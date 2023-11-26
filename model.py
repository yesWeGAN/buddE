import math
import torch
from transformers import DeiTModel
from tokenizer import PatchwiseTokenizer


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

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: Config dict from toml."""
        super().__init__()
        self.model = DeiTModel.from_pretrained(config["encoder"]["pretrained_encoder"])
        self.bottleneck = torch.nn.AdaptiveAvgPool1d(
            int(config["encoder"]["encoder_bottleneck"])
        )

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

    def __init__(self, config: dict, tokenizer: PatchwiseTokenizer) -> None:
        """
        Args:
            config: Dict parsed from toml.
            tokenizer: PatchwiseTokenizer to parse vocab_size, pad_token."""
        super().__init__()
        self.decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=config["decoder"]["decoder_layer_dim"],
            nhead=config["decoder"]["num_heads"],
            batch_first=True
        )
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=config["decoder"]["num_decoder_layers"],
        )
        # the positional embeddings for the decoder
        self.decoder_pos_embed = torch.nn.Parameter(
            torch.randn(
                1,
                int(config["tokenizer"]["max_seq_len"]) - 1,
                int(config["decoder"]["decoder_layer_dim"]),
            )
            # we get a weight shaped 1, max_seq_len, bottleneck_dim. multiply by 0.02 to shrink it?
        )
        self.decoder_pos_drop = torch.nn.Dropout(0.05)

        # the positional embeddings for the encoder [transforms]
        encoder_len = (
            int(config["transforms"]["target_image_size"])
            // int(config["transforms"]["patch_size"])
        ) ** 2
        self.encoder_pos_embed = torch.nn.Parameter(
            torch.randn(
                (1, encoder_len + 2, int(config["encoder"]["encoder_bottleneck"]))
            )
        )
        self.encoder_pos_drop = torch.nn.Dropout(0.05)
        self.vocab_size = int(tokenizer.vocab_size)     # 219
        self.PAD = tokenizer.PAD
        self.EOS = tokenizer.EOS
        self.BOS = tokenizer.BOS
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=int(config["decoder"]["decoder_layer_dim"]),
        )
        self.output = torch.nn.Linear(
            int(config["decoder"]["decoder_layer_dim"]), self.vocab_size
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if name not in ["encoder_pos_embed", "decoder_pos_embed"]:
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
            else:
                print(f"Initializing {name} positional embedding.")
                torch.nn.init.trunc_normal_(p, std=0.02)

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
        y_embed = self.embedding(y) * math.sqrt(self.embedding.embedding_dim)
        # x is the output of the encoder. add positional embeds and dropout
        x = self.encoder_pos_drop(x * math.sqrt(self.embedding.embedding_dim) + self.encoder_pos_embed)
        # y is the input to the decoder. add positional embeds and dropout
        y = self.decoder_pos_drop(y_embed + self.decoder_pos_embed)
        # now both inputs have pos encodings, dropout applied. apply decoder layer to predict
        y_pred = self.decoder(
            tgt=y_embed, memory=x, tgt_mask=y_mask, tgt_key_padding_mask=padding_mask
        )
        # project the outputs into vocab
        outputs = self.output(y_pred)
        return outputs
    
    def predict(self, x: torch.Tensor, y: torch.Tensor)->torch.Tensor:
        """Here will be predict, but first, I need to fix the positional encodings."""


    def mask_tokens(self, y_true: torch.Tensor) -> tuple:
        y_len = y_true.shape[1]  # y_true is shaped B, N, N: max_seq_len
        # create the lower diagonal matrix masking
        y_mask = torch.tril(torch.ones((y_len, y_len), device="cpu"))
        y_mask = (
            y_mask.float()
            .masked_fill(y_mask == 0, float("-inf"))
            .masked_fill(y_mask == 1, float(0.0))
        ).bool().cuda()
        # create a mask for padded tokens
        padding_mask = (y_true == self.PAD).cuda()
        return y_mask, padding_mask


class ODModel(torch.nn.Module):
    def __init__(self, config: dict, tokenizer: PatchwiseTokenizer) -> None:
        """
        Args:
            config: Dict parsed from toml.
            tokenizer: PatchwiseTokenizer to parse vocab_size, pad_token."""
        super().__init__()
        self.encoder = Encoder(config=config)
        self.decoder = Decoder(config=config, tokenizer=tokenizer)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward function for ensemble model.
        Args:
            x: Tensor of shape [BATCH, CHANNELS, IMAGEDIM, IMAGEDIM].
            y: Tensor of shape [BATCH, MAX_SEQ_LEN].
        Returns:
            Tensor of shape [BATCH, MAX_SEQ_LEN]."""

        x_encoded = self.encoder(x)
        preds = self.decoder(x_encoded, y)

        return preds
    
    def predict(self, x: torch.Tensor, max_len: 30) -> torch.Tensor:
        """Predict function for ensemble model.
        Args:
            x: Input from encoder.
            max_len: Maximum sequence length to generate. """
        x_encoded = self.encoder(x)
        preds = torch.ones((x.shape[0],1)).fill_(self.decoder.BOS)
        print(preds)
        print(preds.shape)
        for k in range(max_len):
            pred_step = self.decoder(preds)



# today's learnings:
# positional encodings are learnt Parameters with dropout
# we need to initialize weights. xavier_uniform seems to work for most
# positional encodings are initialized with a truncated distribution
