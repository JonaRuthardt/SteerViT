import torch
import torch.nn as nn
import types
import timm
from steervit.crossattention import GatedCrossAttention, attn_forward_wrapper


def block_forward(self, x):
    """nn.sequential inputs and outputs (x, text_feats, attn_mask) instead of x
    x --> LN1 --> attn -->  LN2 --> FFN --> x
    - injecting cross_attention layer before LN1
    note: both LayerScale and stochastic_depth are Identity()
    """
    x, text_feats, attn_mask = x
    
    if self.gated_cross_attn is not None and text_feats is not None:
        """x --> x + CA(LN(img),text)--> x"""
        x = self.gated_cross_attn(x, text_feats, attn_mask = attn_mask)
        
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

    return (x, text_feats, attn_mask)

def forward_features(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
    if isinstance(x, tuple):
        x, text_feats = x
    else: 
        text_feats = attn_mask = None
    x = self.patch_embed(x)
    x = self._pos_embed(x)
    x = self.patch_drop(x)
    x = self.norm_pre(x)

    x, text_feats, attn_mask = self.blocks((x, text_feats, attn_mask))
    x = self.norm(x)
    return x

###################  Modifications to Timm ViT Backbone  ###################

class ViTBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.resolution = config["resolution"]
        self.trunk = timm.create_model(
            config["model_name"],
            pretrained=True,
            img_size=self.resolution,
        )
        self.trunk.attn_pool = None
        self.trunk.forward_features = types.MethodType(forward_features, self.trunk) #self -> timm.VisionTransformer

        for layer_idx, blk in enumerate(self.trunk.blocks):
            #each block's a diff object
            blk.forward = types.MethodType(block_forward, blk)
            if (layer_idx in config["cross_attn_layers"]):
                blk.gated_cross_attn = GatedCrossAttention(layer_idx, 
                                                        dim=self.trunk.embed_dim,
                                                        ff_mult=config["cross_attn_ffn_mult"],
                                                        use_ffn=config["use_ffn"],
                                                        )
            else:
                blk.gated_cross_attn = None
            blk.attn.forward = types.MethodType(attn_forward_wrapper, blk.attn)

            # Disable fused attention for the last block for visualizing attention maps.
            blk.attn.fused_attn = False if ((layer_idx == len(self.trunk.blocks)-1)) else True
            
    def forward(self, image, text, attn_mask = None):
        return self.trunk.forward_features((image,text), attn_mask = attn_mask)
    