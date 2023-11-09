import torch
from transformers import DeiTModel


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
        self.model = DeiTModel.from_pretrained(config["model"]["pretrained_encoder"])
        self.bottleneck = torch.nn.AdaptiveAvgPool1d(
            int(config["model"]["encoder_bottleneck"])
        )

    def forward(self, x):
        """The forward function. Bottleneck to reduce hidden size dimensionality."""
        hidden_state = self.model(x).last_hidden_state
        return self.bottleneck(hidden_state)


class Decoder(torch.nn.Module):
    """The Decoder consists of num_decoder_layers of nn.TransformerDecoderLayer"""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=config["decoder"]["decoder_layer_dim"],
            nhead=config["decoder"]["num_heads"],
        )
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=config["decoder"]["num_decoder_layers"],
        )
        # the positional embeddings for the decoder
        self.decoder_pos_embed = torch.nn.Parameter(
            torch.randn(
                1,
                int(config["tokenizer"]["max_seq_len"]),
                int(config["decoder"]["decoder_layer_dim"]),
            )
            # we get a weight shaped 1, max_seq_len, bottleneck_dim. multiply by 0.02 to shrink it?
        )
        self.decoder_pos_drop = torch.nn.Dropout(0.05)

        # the positional embeddings for the encoder [transforms]
        encoder_len = (int(config["transforms"]["target_image_size"])//int(config["transforms"]["patch_size"]))**2
        self.encoder_pos_embed = torch.nn.Parameter(torch.randn((1, encoder_len+2, int(config["encoder"]["encoder_bottleneck"]))))
        self.encoder_pos_drop = torch.nn.Dropout(0.05)

        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if name not in ["encoder_pos_embed", "decoder_pos_embed"]:
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
            else:
                print(f"Initializing {name} positional embedding.")
                torch.nn.init.trunc_normal_(p, std=0.02)


# today's learnings:
# positional encodings are learnt Parameters with dropout
# we need to initialize weights. xavier_uniform seems to work for most
# positional encodings are initialized with a truncated distribution


# here a model wrapper for ENCODER and DECODER
