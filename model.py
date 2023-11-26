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
        )
        self.decoder_pos_drop = torch.nn.Dropout(0.05)

        # the positional embeddings for the encoder
        encoder_len = (Config.target_image_size // Config.patch_size) ** 2
        self.encoder_pos_embed = torch.nn.Parameter(
            torch.randn((1, encoder_len + 2, Config.encoder_bottleneck))
        )
        self.encoder_pos_drop = torch.nn.Dropout(0.05)
        self.vocab_size = int(tokenizer.vocab_size)  # 219
        self.PAD = tokenizer.PAD
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
                torch.nn.init.trunc_normal_(p, std=0.02)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Forward call.
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
        """print(
            f"Lets see what we have. \n tgt: {y_embed.shape},\n memory {x.shape}, \n tgt_mask: {y_mask.shape}, \n tgt_key_padding_mask {padding_mask.shape}."
        )
        print(
            f"Additonally, the tensor types should match for: \n tgt_key_padding_mask: {padding_mask.dtype}, \n tgt_mask: {y_mask.dtype}"
        )"""
        y_pred = self.decoder(
            tgt=y_embed, memory=x, tgt_mask=y_mask, tgt_key_padding_mask=padding_mask
        )
        # print(f"After decoder: y_pred shape: {y_pred.shape}")

        """Signature of DecoderLayer forward call:
        forward(
            tgt, # the sequence to the decoder
            memory, # the sequence from the last layer of the encoder 
            tgt_mask=None, 
            memory_mask=None, 
            tgt_key_padding_mask=None, 
            memory_key_padding_mask=None, 
            tgt_is_causal=None, 
            memory_is_causal=False
        )

        tgt: torch.Size([16, 299, 256]),    # shifted right
        memory torch.Size([16, 198, 256]),  # looks right
        tgt_mask: torch.Size([4, 4]),       # says it should be 16x16? -> solved, batch_first=True
        tgt_key_padding_mask torch.Size([16, 299]).
        Additonally, the tensor types should match for: 
        tgt_key_padding_mask: torch.bool, 
        tgt_mask: torch.float32 # solved with .bool()

        After decoder: y_pred shape: torch.Size([16, 299, 256])
        Why does it end up being [16, 219] when criterion is applied?
            torch.Linear expects D_in and D_out to be the last dim

        
    Transformer Layer Shapes:
    https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer


    src: (S,E)(S,E) for unbatched input, (S,N,E)(S,N,E) if batch_first=False or (N, S, E) if batch_first=True.

    tgt: (T,E)(T,E) for unbatched input, (T,N,E)(T,N,E) if batch_first=False or (N, T, E) if batch_first=True.

    src_mask: (S,S)(S,S) or (N⋅num_heads,S,S)(N⋅num_heads,S,S).

    tgt_mask: (T,T)(T,T) or (N⋅num_heads,T,T)(N⋅num_heads,T,T).

    memory_mask: (T,S)(T,S).

    src_key_padding_mask: (S)(S) for unbatched input otherwise (N,S)(N,S).

    tgt_key_padding_mask: (T)(T) for unbatched input otherwise (N,T)(N,T).

    memory_key_padding_mask: (S)(S) for unbatched input otherwise (N,S)(N,S).

    Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked positions. 
    If a BoolTensor is provided, positions with True are not allowed to attend while False values will be unchanged. 
    If a FloatTensor is provided, it will be added to the attention weight. [src/tgt/memory]_key_padding_mask 
    provides specified elements in the key to be ignored by the attention. If a BoolTensor is provided, 
    the positions with the value of True will be ignored while the position with the value of False will be unchanged.



    output: (T,E)(T,E) for unbatched input, (T,N,E)(T,N,E) if batch_first=False or (N, T, E) if batch_first=True.

Note: Due to the multi-head attention architecture in the transformer model, the output sequence length of a transformer is same as the input sequence (i.e. target) length of the decoder.

where S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number
        """
        # project the outputs into vocab
        outputs = self.output(y_pred)
        # print(f"Shape after output projection: {outputs.shape}")
        return outputs

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
            Tensor of shape [BATCH, MAX_SEQ_LEN]."""

        x_encoded = self.encoder(x)
        preds = self.decoder(x_encoded, y)

        return preds


# today's learnings:
# positional encodings are learnt Parameters with dropout
# we need to initialize weights. xavier_uniform seems to work for most
# positional encodings are initialized with a truncated distribution

# today's learnings:
# positional encodings in ViT are not sin/cos, but they are learnt
# positional encodings should be scaled down so they don't overwhelm the actual embeddings
