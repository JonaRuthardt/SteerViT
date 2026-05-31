import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import hf_hub_download
from timm.data import resolve_data_config, create_transform

from steervit.backbone import ViTBackbone
from steervit.utils import TCAttentionExtract

class SteerViT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self._device = torch.device("cpu")
        self.attention_extractor = None
        self._gate_params = {}
        self._gate_factor = 1.0

        #### Load Vision Model ####
        self.vision_model = ViTBackbone(config["vision_encoder"])
        assert self.image_size[0] % self.patch_size == 0, "Image resolution must be divisible by patch size"
        self.num_img_tokens = (self.image_size[0] // self.patch_size) ** 2 + self.vision_model.trunk.num_prefix_tokens
        self.feature_aggregation = config["vision_encoder"]["feature_aggregation"] # "cls" or "mean"
        assert self.feature_aggregation in ["cls", "mean"], "Feature aggregation must be either 'cls' or 'mean'"
        if self.feature_aggregation == "cls":
            assert self.vision_model.trunk.num_prefix_tokens > 0, "Model must have a cls token for cls feature aggregation"

        self.visual_dim = self.vision_model.trunk.embed_dim

        #### Load Language Model ####
        text_encoder = config['text_encoder']
        if "roberta" in text_encoder.lower():
            from transformers import RobertaTokenizer, RobertaModel
            self.tokenizer = RobertaTokenizer.from_pretrained(text_encoder)
            self.text_model = RobertaModel.from_pretrained(text_encoder).eval()
            self.text_dim = self.text_model.config.hidden_size
        else:
            raise NotImplementedError(f"Text encoder {text_encoder} currently not implemented")
        for p in self.text_model.parameters():
            p.requires_grad = False

        #### Load Language-Image Connector ####
        self.connector = Connector(self.text_dim, self.visual_dim)

        #### Load Segmentation Head ####
        self.lin_seg_head = nn.Linear(self.visual_dim, 1, bias = True)
        nn.init.constant_(self.lin_seg_head.weight, 0)
        nn.init.constant_(self.lin_seg_head.bias, 0)

    def to(self, device):
        # super().to(device)
        self.text_model = self.text_model.to(device)
        self.vision_model = self.vision_model.to(device)
        self.connector = self.connector.to(device)
        self.lin_seg_head = self.lin_seg_head.to(device)
        self._device = device
        return self

    @classmethod
    def from_pretrained(cls, checkpoint_name, device=None):
        if os.path.isfile(checkpoint_name):
            ckpt_path = checkpoint_name
        else:
            ckpt_path = hf_hub_download(
                repo_id="JonaRuthardt/SteerViT",
                filename=checkpoint_name,
            )

        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )

        model = cls(checkpoint["config"])
        model.load_state_dict(
            checkpoint["state_dict"],
            strict=False,
        )
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        if device is not None:
            model = model.to(device)
        return model

    @property
    def patch_size(self):
        return self.vision_model.trunk.patch_embed.patch_size[0]

    @property
    def feature_dim(self):
        return self.vision_model.trunk.embed_dim

    @property
    def image_size(self):
        return (self.vision_model.resolution, self.vision_model.resolution)

    def get_transforms(self):
        vision_config = resolve_data_config({}, model=self.vision_model.trunk)
        vision_config["input_size"] = (3, self.image_size[0], self.image_size[1])
        transform = create_transform(**vision_config)
        return transform

    def forward(self, images: torch.Tensor, texts: list[str] = None, return_segmentation_logits=False):
        if texts is not None:
            # Text conditioning
            assert images.size(0) == len(texts), "Batch size of images and texts must match"

            roberta_dict = self.tokenizer(texts, padding=True, truncation=True,max_length=512, return_tensors='pt')
            roberta_dict = {k: v.to(self.text_model.device) for k, v in roberta_dict.items()}
            text_feats = self.text_model(**roberta_dict).last_hidden_state
            attn_mask = roberta_dict['attention_mask'].bool()

            text_feats = F.normalize(text_feats, dim=-1) #precomputed text feats
            text_feats = self.connector(text_feats)

            pos_ids = torch.arange(text_feats.size(1), dtype = torch.long, device = text_feats.device)
            pos_ids = pos_ids.unsqueeze(0).expand(text_feats.shape[0], -1)
            attn_mask = torch.cat((torch.ones(text_feats.size(0), self.num_img_tokens).bool().to(attn_mask.device), attn_mask), dim= -1)
        else:
            # Equivalent to vanilla base ViT model
            text_feats = attn_mask = None

        img_feats = self.vision_model(images, text_feats, attn_mask = attn_mask) #text conditioned img feats

        if return_segmentation_logits:
            return self.get_heatmap_logits(img_feats)
        return img_feats

    def get_heatmap_logits(self, img_feats):
        return self.lin_seg_head(img_feats[:, self.vision_model.trunk.num_prefix_tokens:, :]).squeeze(-1)

    @torch.no_grad()
    def get_dense_features(self, images: torch.Tensor, texts: list[str] = None):
        return self.forward(images.to(self._device), texts)[:, self.vision_model.trunk.num_prefix_tokens:, :]

    @torch.no_grad()
    def get_global_features(self, images: torch.Tensor, texts: list[str] = None):
        feats = self.forward(images.to(self._device), texts)
        if self.feature_aggregation == 'cls':
            assert self.vision_model.trunk.num_prefix_tokens > 0, "Model must have a cls token for cls feature aggregation"
            return feats[:, 0, :]
        elif self.feature_aggregation == 'mean':
            return torch.mean(feats[:, self.vision_model.trunk.num_prefix_tokens:, :], dim=1)
        else:
            raise NotImplementedError(f"Feature aggregation {self.feature_aggregation} not implemented")

    @torch.no_grad()
    def get_heatmaps(self, images: torch.Tensor, texts: list[str] = None):
        heatmap_logits = self.get_heatmap_logits(self.forward(images.to(self._device), texts))
        heatmaps = F.softmax(heatmap_logits, dim=1).view(images.size(0), 1, self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size)
        heatmaps = F.interpolate(heatmaps, size=self.image_size, mode='bilinear', align_corners=False)
        return heatmaps

    @torch.no_grad()
    def get_attention_heatmaps(self, images: torch.Tensor, texts: list[str] = None, **kwargs):
        if self.attention_extractor is None:
            self.attention_extractor = TCAttentionExtract(
                model=self,
                mode='eval',
                method='hook',
            )

        heatmaps = self.attention_extractor.get_attention_heatmaps(
            imgs=images.to(self._device), texts=texts, num_prefix_tokens=self.vision_model.trunk.num_prefix_tokens, **kwargs)

        return heatmaps

    def set_gate_factor(self, factor: float):
        for blk_idx, blk in enumerate(self.vision_model.trunk.blocks):
            gca = getattr(blk, "gated_cross_attn", None)
            if gca is not None:
                with torch.no_grad():
                    if blk_idx not in self._gate_params:
                        # Store original gate parameters
                        self._gate_params[blk_idx] = {
                            "attn_gate": gca.attn_gate.clone(),
                            "ff_gate": gca.ff_gate.clone() if hasattr(gca, "ff_gate") else None,
                        }
                    gca.attn_gate.copy_(self._gate_params[blk_idx]["attn_gate"] * float(factor))
                    if hasattr(gca, "ff_gate"):
                        gca.ff_gate.copy_(self._gate_params[blk_idx]["ff_gate"] * float(factor))
        self._gate_factor = factor

class Connector(nn.Module):
    def __init__(self, input_dim, output_dim=1152):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim, bias=True)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)