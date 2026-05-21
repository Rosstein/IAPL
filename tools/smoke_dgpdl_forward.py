import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import models.clip_models as clip_models


class FakeVisualEncoder(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.output_dim = feature_dim

    def forward(self, image, shared_ctx, compound_deeper_prompts):
        pooled = image.mean(dim=(2, 3))
        if pooled.shape[1] < self.output_dim:
            pooled = torch.nn.functional.pad(pooled, (0, self.output_dim - pooled.shape[1]))
        return pooled[:, :self.output_dim], []


class FakeClip(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.visual = FakeVisualEncoder(feature_dim)
        self.dtype = torch.float32


def build_args():
    return SimpleNamespace(
        n_ctx=2,
        prompt_depth=9,
        image_size=224,
        vision_width=1024,
        clip_path="unused.pt",
        vit_adapter_list=[],
        text_adapter_list=[],
        gate=False,
        tta=False,
        condition=True,
        cond_type="dgpdl",
        cond_dim=512,
        cond_scale_init=1e-6,
        feature_dim=None,
        loss_adapter=1.0,
        loss_contrast=1.0,
        loss_condition=1.0,
        use_contrast=False,
        smooth=False,
    )


def main():
    args = build_args()
    feature_dim = 768
    original_loader = clip_models.load_clip_to_cpu
    clip_models.load_clip_to_cpu = lambda *loader_args, **loader_kwargs: FakeClip(feature_dim)
    try:
        model = clip_models.CLIPModel(args)
        model.train()

        image = torch.randn(2, 3, args.image_size, args.image_size)
        logits, image_features, pred_bias = model(image)
        bias, _ = model.conditional_ctx(image)

        print(f"logits shape: {tuple(logits.shape)}")
        print(f"image_features shape: {tuple(image_features.shape)}")
        print(f"bias shape: {tuple(bias.shape)}")
        print(f"pred_bias shape: {tuple(pred_bias.shape)}")

        assert logits.shape == (2, 1)
        assert image_features.shape == (2, feature_dim)
        assert bias.shape == (2, args.n_ctx, args.vision_width)
        assert pred_bias.shape == (2, 1)
    finally:
        clip_models.load_clip_to_cpu = original_loader


if __name__ == "__main__":
    main()
