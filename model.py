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
        self.model = DeiTModel.from_pretrained(config["model"]["pretrained_encoder"])
        self.bottleneck = torch.nn.AdaptiveAvgPool1d(
            int(config["model"]["encoder_bottleneck"])
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
        self.vocab_size = tokenizer.vocab_size
        self.PAD = tokenizer.PAD
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size, 
            embedding_dim=config["decoder"]["decoder_layer_dim"]
            )
        self.output = torch.nn.Linear(int(config["decoder"]["decoder_layer_dim"]), self.vocab_size)

        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if name not in ["encoder_pos_embed", "decoder_pos_embed"]:
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
            else:
                print(f"Initializing {name} positional embedding.")
                torch.nn.init.trunc_normal_(p, std=0.02)


    def forward(self, x: torch.Tensor, y:torch.Tensor):
        """Forward call.
        Args:
            x: Encoder ouput. Tensor of shape [BATCH, NUM_PATCHES, BOTTLENECK_DIM]
            y: Token sequence. Tensor of shape [BATCH, MAX_SEQ_LEN]
        Returns:
            Tensor. TODO: what shape does it have, MAX_SEQ_LEN?."""

        # create the masks for the truth
        y_mask, padding_mask = self.mask_tokens(y)
        # project the truth with embedding layer
        y_embed = self.embedding(y)
        # x is the output of the encoder. add positional embeds and dropout
        x = self.encoder_pos_drop(x + self.encoder_pos_embed)
        # y is the input to the decoder. add positional embeds and dropout
        y = self.decoder_pos_drop(y_embed + self.decoder_pos_embed)
        # now both inputs have pos encodings, dropout applied. apply decoder layer to predict
        # TODO: could be that right here, I need to transpose x and y 
        y_pred = self.decoder(tgt=y_embed,
                              memory=x,
                              tgt_mask=y_mask,
                              tgt_key_padding_mask=padding_mask)
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
        """
        # project the outputs into vocab
        return self.output(y_pred)
    
    def mask_tokens(self, y_true: torch.Tensor)-> tuple:
        y_len = y_true.shape[1] # y_true is shaped B, N, N: max_seq_len
        # create the lower diagonal matrix masking 
        y_len = 4
        y_mask = torch.tril(torch.ones((y_len, y_len), device='cpu'))
        y_mask = y_mask.float().masked_fill(y_mask==0,float('-inf')).masked_fill(y_mask==1, float(0.0))
        # create a mask for padded tokens
        padding_mask = (y_true == self.PAD)
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
    
    def forward(self, x: torch.Tensor, y: torch.Tensor)->torch.Tensor:
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