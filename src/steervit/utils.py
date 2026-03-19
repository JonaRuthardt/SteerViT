from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from timm.utils import AttentionExtract

class TCAttentionExtract(AttentionExtract):
    """ Attention extractor that returns attention heatmaps averaged over heads. """
    @torch.no_grad()
    def forward(self, imgs: torch.Tensor, texts: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        if texts is None:
            output = self.model(imgs) # Non-TC model
        else: 
            output = self.model(imgs, texts)
        if self.hooks is not None:
            output = self.hooks.get_output(device=imgs.device)
        return output
    
    @torch.no_grad()
    def get_attention_heatmaps(self, imgs: torch.Tensor, texts: List[str] = None, layer: int = -1, num_prefix_tokens: int = 1, head_pooling: str = "mean", remove_sinks: bool = True, interpolate: bool = True) -> Dict[str, torch.Tensor]:
        attention_maps = self.forward(imgs, texts) # dict of attention maps from hooks or fx
        attn_layer_list = list(attention_maps.values())
        attention_map = attn_layer_list[layer]  # (B, num_heads, N, N)
        cls_attention_map = attention_map[:, :, 0, num_prefix_tokens:]  # CLS token to all (non-special) patches
        
        B, num_heads, N = cls_attention_map.shape
        h = w = int(N ** 0.5)
        H, W = imgs.shape[-2], imgs.shape[-1]

        cls_attention_map = cls_attention_map.reshape(B, num_heads, h, w)
        if head_pooling == "mean":
            heatmaps = cls_attention_map.mean(dim=1)  # (B, h, w)
        elif head_pooling == "max":
            heatmaps, _ = cls_attention_map.max(dim=1)  # (B, h, w)
        elif head_pooling == "min":
            heatmaps, _ = cls_attention_map.min(dim=1)  # (B, h, w)
        elif head_pooling == "median":
            heatmaps = cls_attention_map.median(dim=1).values  # (B, h, w)
        elif head_pooling == "mean_std":
            heatmaps = cls_attention_map.mean(dim=1) - cls_attention_map.std(dim=1)  # (B, h, w)
        elif head_pooling == "none":
            return cls_attention_map# return all heads separately
        else:
            raise ValueError(f"Unknown head_pooling method: {head_pooling}")
        
        if remove_sinks:
            heatmaps, _ = suppress_attention_sinks_multi(heatmaps)

        heatmaps = (heatmaps - heatmaps.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values) / (heatmaps.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values - heatmaps.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values + 1e-6)  # normalize to [0, 1]

        if interpolate:
            heatmaps = F.interpolate(heatmaps.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)  # (B, H, W)
            
        return heatmaps
    

def _local_mean_excluding_center(x: torch.Tensor, radius: int, fallback: torch.Tensor) -> torch.Tensor:
    """
    x:        (N, H, W)
    fallback: (N,) global medians, used only for degenerate 1x1 maps
    returns:  (N, H, W)
    """
    n, h, w = x.shape
    k = 2 * radius + 1

    x4 = x[:, None, :, :]  # (N, 1, H, W)

    kernel = torch.ones((1, 1, k, k), dtype=x.dtype, device=x.device)
    kernel[0, 0, radius, radius] = 0.0  # exclude center pixel

    # Sum of neighbors
    sums = F.conv2d(x4, kernel, padding=radius)

    # Number of valid neighbors at each location
    counts = F.conv2d(torch.ones_like(x4), kernel, padding=radius)

    local_means = sums / counts.clamp_min(1.0)
    local_means = torch.where(
        counts > 0,
        local_means,
        fallback.view(n, 1, 1, 1),
    )

    return local_means[:, 0]  # (N, H, W)


def suppress_attention_sinks(
    heatmaps,
    threshold_sigma=5.0,
    local_radius=None,
    min_ratio=None,
    replace_with="zero",   # "zero", "global_mean", "local_mean"
):
    """
    Suppress attention sinks in a heatmap or batch of heatmaps.

    Args:
        heatmaps: tensor/array with shape (..., H, W)
                  e.g. (21, 21) or (64, 21, 21)
        threshold_sigma: threshold = median + sigma * robust_std
        local_radius: optional neighborhood radius
        min_ratio: optional additional constraint:
                   x > min_ratio * reference
        replace_with: one of {"zero", "global_mean", "local_mean"}

    Returns:
        cleaned: same shape as input
        sink_mask: same shape as input, bool
    """
    x = torch.as_tensor(heatmaps)

    if x.ndim < 2:
        raise ValueError("heatmaps must have shape (..., H, W)")

    if not torch.is_floating_point(x):
        x = x.float()

    *batch_shape, h, w = x.shape
    x = x.reshape(-1, h, w)   # (N, H, W)

    flat = x.flatten(1)       # (N, H*W)

    # Robust global threshold: median + k * MAD
    med = flat.median(dim=1).values
    mad = (flat - med[:, None]).abs().median(dim=1).values + 1e-8
    robust_std = 1.4826 * mad
    global_thresh = med + threshold_sigma * robust_std

    sink_mask = x > global_thresh[:, None, None]

    local_means = None
    if local_radius is not None:
        if local_radius < 1:
            raise ValueError("local_radius must be >= 1")
        local_means = _local_mean_excluding_center(x, local_radius, med)

        # Same logic as your original code
        sink_mask &= x > local_means

    if min_ratio is not None:
        if local_means is not None:
            ref = local_means
        else:
            ref = flat.mean(dim=1).view(-1, 1, 1)
        sink_mask &= x > (min_ratio * (ref + 1e-8))

    # Replacement
    if replace_with == "zero":
        replacement = torch.zeros_like(x)
    elif replace_with == "global_mean":
        replacement = flat.mean(dim=1).view(-1, 1, 1).expand_as(x)
    elif replace_with == "local_mean":
        if local_means is None:
            raise ValueError("replace_with='local_mean' requires local_radius != None")
        replacement = local_means
    else:
        raise ValueError("replace_with must be one of: zero, global_mean, local_mean")

    cleaned = torch.where(sink_mask, replacement, x)

    cleaned = cleaned.reshape(*batch_shape, h, w)
    sink_mask = sink_mask.reshape(*batch_shape, h, w)

    return cleaned, sink_mask

def suppress_attention_sinks_multi(
    heatmaps,
    passes=None,
    return_all_masks=False,
):
    """
    Apply multiple sink-suppression passes.

    Args:
        heatmaps: (..., H, W)
        passes: list of dicts, each passed to suppress_attention_sinks
        return_all_masks: if True, also return per-pass masks

    Returns:
        cleaned: final cleaned heatmaps
        combined_mask: union of sink masks across passes
        all_masks: list of per-pass masks (optional)
    """
    if passes is None:
        passes = [
            dict(threshold_sigma=5.0, local_radius=1, min_ratio=5.0, replace_with="global_mean"),
            dict(threshold_sigma=5.0, local_radius=1, min_ratio=5.0, replace_with="global_mean"),
            dict(threshold_sigma=5.0, local_radius=1, min_ratio=5.0, replace_with="local_mean"),
        ]

    cleaned = heatmaps
    combined_mask = None
    all_masks = []

    for cfg in passes:
        cleaned, mask = suppress_attention_sinks(cleaned, **cfg)
        combined_mask = mask if combined_mask is None else (combined_mask | mask)
        all_masks.append(mask)

    if return_all_masks:
        return cleaned, combined_mask, all_masks
    return cleaned, combined_mask