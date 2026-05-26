import torch
import torch.nn as nn


class AttentionBase2Map(nn.Module):
    def __init__(self, dim, qkv_bias=False):
        super().__init__()
        self.scale = dim ** -0.5
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, query, bases):
        batch_size, channels = query.shape
        bases = bases.unsqueeze(0).expand(batch_size, -1, -1)
        q = self.wq(query).unsqueeze(1)
        k = self.wk(bases)
        v = self.wv(bases)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        mapped = attn @ v
        return mapped, attn


class DGPDL_Condition_Module(nn.Module):
    def __init__(
        self,
        n_ctx=2,
        vision_width=1024,
        cond_dim=None,
        scale_init=1e-6,
        n_bases=8,
        patch_size=16,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.vision_width = vision_width
        self.patch_size = patch_size

        embed_dim = cond_dim or vision_width
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.token_norm = nn.LayerNorm(embed_dim)

        bases_mu = torch.empty(n_bases, embed_dim)
        nn.init.normal_(bases_mu, std=0.02)
        self.bases_mu = nn.Parameter(bases_mu)

        bases_sig = torch.empty(n_bases, embed_dim)
        nn.init.normal_(bases_sig, std=0.02)
        self.bases_sig = nn.Parameter(bases_sig)

        self.attn_mu = AttentionBase2Map(embed_dim, qkv_bias=True)
        self.attn_sig = AttentionBase2Map(embed_dim, qkv_bias=True)

        if embed_dim != vision_width:
            self.project_to_vision = nn.Linear(embed_dim, vision_width)
        else:
            self.project_to_vision = None

        if n_ctx == 2:
            self.bias_head = None
        else:
            self.bias_head = nn.Linear(2 * vision_width, n_ctx * vision_width)

        self.pred_head = nn.Sequential(
            nn.LayerNorm(2 * vision_width),
            nn.Linear(2 * vision_width, 1),
        )
        self.scale = nn.Parameter(torch.full((1, n_ctx, vision_width), scale_init))

    def _stats(self, tokens, eps=1e-6):
        mu = tokens.mean(dim=1)
        var = tokens.var(dim=1, unbiased=False)
        sig = (var + eps).sqrt()
        return mu, sig

    def forward(self, x):
        batch_size = x.shape[0]
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = self.token_norm(tokens)

        mu, sig = self._stats(tokens)
        map_mu, _ = self.attn_mu(mu, self.bases_mu)
        map_sig, _ = self.attn_sig(sig, self.bases_sig)

        if self.project_to_vision is not None:
            map_mu = self.project_to_vision(map_mu)
            map_sig = self.project_to_vision(map_sig)
            mu = self.project_to_vision(mu)
            sig = self.project_to_vision(sig)

        ds_prompt_vis = torch.cat([map_mu, map_sig], dim=1)

        if self.bias_head is None:
            bias = ds_prompt_vis
        else:
            bias = self.bias_head(ds_prompt_vis.flatten(1)).view(batch_size, self.n_ctx, self.vision_width)

        bias = bias * self.scale
        pred_bias = self.pred_head(torch.cat([mu, sig], dim=1))

        return bias, pred_bias
