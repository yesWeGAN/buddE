import math
import torch
from transformers import DeiTModel
from tokenizer import PatchwiseTokenizer
from config import Config


class Encoder(torch.nn.Module):
    """
    Summary:
    model has DeiTEncoder + embedding layer
    embeddinger layer: DeiTEmbeddings, which uses DeiTPatchEmbeddings:
        a Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16)) projecting as follows:
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    which essentially means:
    the tokenization of the image is a learnt convolutional operation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = DeiTModel.from_pretrained(Config.pretrained_encoder)
        self.bottleneck = torch.nn.AdaptiveAvgPool1d(Config.encoder_bottleneck)

    def forward(self, x: torch.Tensor):
        """The forward function. Bottleneck to reduce hidden size dimensionality.
        Args:
            x: Tensor of shape [BATCH, CHANNELS, IMAGEDIM, IMAGEDIM]

        Returns:
            Tensor of shape [BATCH, NUM_PATCHES, BOTTLENECK_DIM]."""
        hidden_state = self.model(x).last_hidden_state
        return self.bottleneck(hidden_state)


class Decoder(torch.nn.Module):
    """The Decoder consists of num_decoder_layers of nn.TransformerDecoderLayer"""

    def __init__(self, tokenizer: PatchwiseTokenizer) -> None:
        """
        Args:
            tokenizer: PatchwiseTokenizer to parse vocab_size, pad_token."""
        super().__init__()
        self.decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=Config.decoder_layer_dim,
            nhead=Config.num_heads,
            batch_first=True,
        )
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=Config.num_decoder_layers,
        )
        # the positional embeddings for the decoder. in 2D, they are not sin/cos, but learnt.
        self.decoder_pos_embed = torch.nn.Parameter(
            torch.randn(
                1,
                Config.max_seq_len - 1,
                Config.decoder_layer_dim,
            )
            * (1 / math.sqrt(Config.decoder_layer_dim))
        )
        self.decoder_pos_drop = torch.nn.Dropout(Config.dropout)

        # the positional embeddings for the encoder
        encoder_len = (Config.target_image_size // Config.patch_size) ** 2
        self.encoder_pos_embed = torch.nn.Parameter(
            torch.randn((1, encoder_len + 2, Config.encoder_bottleneck))
            * (1 / math.sqrt(Config.decoder_layer_dim))
        )
        self.encoder_pos_drop = torch.nn.Dropout(Config.dropout)
        self.vocab_size = int(tokenizer.vocab_size)  # 219
        self.PAD = tokenizer.PAD
        self.EOS = tokenizer.EOS
        self.BOS = tokenizer.BOS
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=Config.decoder_layer_dim,
        )
        self.output = torch.nn.Linear(Config.decoder_layer_dim, self.vocab_size)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if name not in ["encoder_pos_embed", "decoder_pos_embed"]:
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.trunc_normal_(p, std=(1 / math.sqrt(Config.decoder_layer_dim)))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Forward call.
        Transformer Layer Shapes:
        https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
        Args:
            x: Encoder ouput. Tensor of shape [BATCH, NUM_PATCHES, BOTTLENECK_DIM]
            y: Token sequence. Tensor of shape [BATCH, MAX_SEQ_LEN-1]
        Returns:
            Tensor. torch.Size([BATCH, MAX_SEQ_LEN-1, BOTTLENECK_DIM])"""

        # create the masks for the truth
        y_mask, padding_mask = self.mask_tokens(y)
        # project the truth with embedding layer
        y_embed = self.embedding(y)
        # x is the output of the encoder. add positional embeds and dropout
        x = self.encoder_pos_drop(x + self.encoder_pos_embed)
        # y is the input to the decoder. add positional embeds and dropout
        y = self.decoder_pos_drop(y_embed + self.decoder_pos_embed)
        # now both inputs have pos encodings, dropout applied. apply decoder layer to predict

        y_pred = self.decoder(
            tgt=y_embed, memory=x, tgt_mask=y_mask, tgt_key_padding_mask=padding_mask
        )
        # project the outputs into vocab
        outputs = self.output(y_pred)
        return outputs

    def predict(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Predict predicts the next token, given the inputs."""
        # pad what has been predicted already to the max_seq_len
        batch_size, y_len = y.shape
        print(f"Last target tokens: {y[:,-1]}")
        padded_y = torch.ones((batch_size, Config.max_seq_len-y_len-1)).fill_(self.PAD).long().to(Config.device)
        padded_y = torch.cat([y, padded_y], dim=1)
        # this part is the same as for forward(). pos_drop should be disabled with eval()
        y_mask, padding_mask = self.mask_tokens(padded_y)
        y_embed = self.embedding(y)
        x = self.encoder_pos_drop(x + self.encoder_pos_embed)
        y = self.decoder_pos_drop(y_embed + self.decoder_pos_embed)
        y_pred = self.decoder(
            tgt=y_embed, memory=x, tgt_mask=y_mask, tgt_key_padding_mask=padding_mask
        )
        outputs = self.output(y_pred)
        print(f"Predicted tokens at y_len-2: \n{outputs[:,y_len-2,:]}\n")
        print(f"Predicted tokens at y_len-1: \n{outputs[:,y_len-1,:]}\n")
        print(f"Predicted tokens at y_len: \n{outputs[:,y_len,:]}\n")
        return outputs[:,y_len,:]

    def mask_tokens(self, y_true: torch.Tensor) -> tuple:
        y_len = y_true.shape[1]  # y_true is shaped B, N, N: max_seq_len
        # create the lower diagonal matrix masking
        y_mask = torch.tril(torch.ones((y_len, y_len), device="cpu"))
        y_mask = (
            (
                y_mask.float()
                .masked_fill(y_mask == 0, float("-inf"))
                .masked_fill(y_mask == 1, float(0.0))
            )
            .bool()
            .cuda()
        )
        # create a mask for padded tokens
        padding_mask = (y_true == self.PAD).cuda()
        return y_mask, padding_mask


class ODModel(torch.nn.Module):
    def __init__(self, tokenizer: PatchwiseTokenizer) -> None:
        """
        Args:
            config: Dict parsed from toml.
            tokenizer: PatchwiseTokenizer to parse vocab_size, pad_token."""
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(tokenizer=tokenizer)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward function for ensemble model.
        Args:
            x: Tensor of shape [BATCH, CHANNELS, IMAGEDIM, IMAGEDIM].
            y: Tensor of shape [BATCH, MAX_SEQ_LEN].
        Returns:
            Tensor of shape [BATCH, MAX_SEQ_LEN, VOCAB_SIZE]."""

        x_encoded = self.encoder(x)
        preds = self.decoder(x_encoded, y)

        return preds

    def predict(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Predict function for ensemble model.
        Args:
            x: Input from encoder.
            y: Tokens generated so far."""
        x_encoded = self.encoder(x)
        y_pred = self.decoder(x_encoded, y)
        return y_pred


# today's learnings:
# positional encodings are learnt Parameters with dropout
# we need to initialize weights. xavier_uniform seems to work for most
# positional encodings are initialized with a truncated distribution

# today's learnings:
# positional encodings in ViT are not sin/cos, but they are learnt
# positional encodings should be scaled down so they don't overwhelm the actual embeddings

# today's learnings: 
# augmentations delay overfitting and reduce validation batch variability.
