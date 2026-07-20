"""Export LightGlue's SIFT-trained matcher to ONNX as a standalone module.

Self-contained (no dependency on lightglue_dynamo's package __init__, which
pulls in torchvision via its aliked.py just to expose unrelated classes).
Classes below are copied from fabio-sim/LightGlue-ONNX's
lightglue_dynamo/models/lightglue.py (a static-graph, ONNX-export-ready
reimplementation of cvg/LightGlue), with one change: the positional
encoding's input dimension M is made configurable (4 instead of a hardcoded
2) to support SIFT's add_scale_ori=True config (keypoints packed as
x,y,scale,orientation instead of just x,y -- see cvg/LightGlue's
lightglue.py L368-372, `"sift": {"input_dim": 128, "add_scale_ori": True}`).

No official ONNX export exists for SIFT+LightGlue anywhere (checked every
release of fabio-sim/LightGlue-ONNX: only superpoint/disk/raco_aliked are
supported) -- this fills that gap for a project that keeps its own SIFT
extractor and only wants to swap the matcher.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def multi_head_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, n, d = q.shape
    head_dim = d // num_heads
    q, k, v = (t.reshape((b, n, num_heads, head_dim)).transpose(1, 2) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape((b, n, d))


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, descriptor_dim: int, num_heads: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = descriptor_dim // num_heads
        self.Wr = nn.Linear(M, head_dim // 2, bias=False)
        self.gamma = gamma
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines]).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0, desc1):
        return (self.token(desc0.detach()).squeeze(-1), self.token(desc1.detach()).squeeze(-1))


class SelfBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x, encoding):
        b, n, _ = x.shape
        qkv = self.Wqkv(x)
        qkv = qkv.reshape((b, n, self.num_heads, self.head_dim, 3)).transpose(1, 2)
        qk, v = qkv[..., :2], qkv[..., 2]
        qk = self.apply_cached_rotary_emb(encoding, qk)
        q, k = qk[..., 0], qk[..., 1]
        q, k, v = (t.transpose(1, 2).reshape(b, n, self.embed_dim) for t in (q, k, v))
        context = multi_head_attention(q, k, v, self.num_heads)
        message = self.out_proj(context)
        return x + self.ffn(torch.concat([x, message], 2))

    def rotate_half(self, qk):
        qk = qk.unflatten(-2, (-1, 2))
        qk = torch.stack((-qk[..., 1, :], qk[..., 0, :]), dim=-2)
        return qk.flatten(-3, -2)

    def apply_cached_rotary_emb(self, encoding, qk):
        return qk * encoding[0].unsqueeze(-1) + self.rotate_half(qk) * encoding[1].unsqueeze(-1)


class CrossBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_out = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, descriptors):
        qk, v = self.to_qk(descriptors), self.to_v(descriptors)
        point_count = descriptors.shape[1]
        qk_swapped = qk.reshape(-1, 2, point_count, self.embed_dim).flip(1).flatten(0, 1)
        v_swapped = v.reshape(-1, 2, point_count, self.embed_dim).flip(1).flatten(0, 1)
        m = multi_head_attention(qk, qk_swapped, v_swapped, self.num_heads)
        m = self.to_out(m)
        descriptors = descriptors + self.ffn(torch.concat([descriptors, m], 2))
        return descriptors


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.self_attn = SelfBlock(embed_dim, num_heads)
        self.cross_attn = CrossBlock(embed_dim, num_heads)

    def forward(self, descriptors, encodings):
        descriptors = self.self_attn(descriptors, encodings)
        return self.cross_attn(descriptors)


def sigmoid_log_double_softmax(similarities, z):
    certainties = F.logsigmoid(z[0::2]) + F.logsigmoid(z[1::2]).transpose(1, 2)
    scores0 = F.log_softmax(similarities, 2)
    scores1 = F.log_softmax(similarities, 1)
    scores = scores0 + scores1 + certainties
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.25
        self.final_proj = nn.Linear(dim, dim, bias=True)
        self.matchability = nn.Linear(dim, 1, bias=True)

    def forward(self, descriptors):
        mdescriptors = self.final_proj(descriptors) / self.scale
        similarities = mdescriptors[0::2] @ mdescriptors[1::2].transpose(1, 2)
        z = self.matchability(descriptors)
        scores = sigmoid_log_double_softmax(similarities, z)
        return scores

    def get_matchability(self, desc):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores, threshold: float):
    max0 = scores.max(2)
    max1 = scores.max(1)
    m0, m1 = max0.indices, max1.indices
    indices = torch.arange(m0.shape[1], device=m0.device).expand_as(m0)
    mutual = indices == m1.gather(1, m0)
    mscores = max0.values.exp()
    valid = mscores > threshold
    b_idx, m0_idx = torch.where(valid & mutual)
    m1_idx = m0[b_idx, m0_idx]
    matches = torch.concat([b_idx[:, None], m0_idx[:, None], m1_idx[:, None]], 1)
    mscores = mscores[b_idx, m0_idx]
    return matches, mscores


class LightGlueSift(nn.Module):
    """Static-graph LightGlue matcher for SIFT descriptors (fixed depth,
    no adaptive early-stopping/pruning -- required for a clean ONNX export,
    same tradeoff fabio-sim's exporter makes for the other extractors)."""

    def __init__(
        self,
        url: str,
        input_dim: int = 128,
        descriptor_dim: int = 256,
        num_heads: int = 4,
        n_layers: int = 9,
        filter_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.filter_threshold = filter_threshold

        if input_dim != descriptor_dim:
            self.input_proj = nn.Linear(input_dim, descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        # add_scale_ori=True -> posenc input dim = 2 + 2*1 = 4 (x,y,scale,ori)
        self.posenc = LearnableFourierPositionalEncoding(4, descriptor_dim, num_heads)

        d, h, n = descriptor_dim, num_heads, n_layers
        self.transformers = nn.ModuleList([TransformerLayer(d, h) for _ in range(n)])
        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList([TokenConfidence(d) for _ in range(n - 1)])

        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", weights_only=True)
        for i in range(n):
            pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"[load_state_dict] missing={missing}")
        print(f"[load_state_dict] unexpected={unexpected}")
        assert "posenc.Wr.weight" not in missing, "posenc.Wr did not load from checkpoint!"

    def forward(self, keypoints: torch.Tensor, descriptors: torch.Tensor):
        # keypoints: (2B, N, 4) = normalized x,y + raw scale,ori
        # descriptors: (2B, N, 128) = RootSIFT
        descriptors = self.input_proj(descriptors)
        encodings = self.posenc(keypoints)
        for i in range(self.n_layers):
            descriptors = self.transformers[i](descriptors, encodings)
        scores = self.log_assignment[i](descriptors)
        matches, mscores = filter_matches(scores, self.filter_threshold)
        return matches, mscores


def main():
    url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/sift_lightglue.pth"
    model = LightGlueSift(url).eval()

    N0, N1 = 500, 480
    N = max(N0, N1)
    kpts0 = F.pad(torch.randn(1, N0, 4), (0, 0, 0, N - N0))
    kpts1 = F.pad(torch.randn(1, N1, 4), (0, 0, 0, N - N1))
    desc0 = F.pad(F.normalize(torch.randn(1, N0, 128), dim=-1), (0, 0, 0, N - N0))
    desc1 = F.pad(F.normalize(torch.randn(1, N1, 128), dim=-1), (0, 0, 0, N - N1))
    keypoints = torch.cat([kpts0, kpts1], dim=0)
    descriptors = torch.cat([desc0, desc1], dim=0)

    with torch.no_grad():
        matches, mscores = model(keypoints, descriptors)
    print("dummy forward OK, matches shape:", matches.shape, "mscores shape:", mscores.shape)

    out_path = "/tmp/lightglue_sift.onnx"
    torch.onnx.export(
        model,
        (keypoints, descriptors),
        out_path,
        input_names=["keypoints", "descriptors"],
        output_names=["matches", "mscores"],
        dynamic_axes={
            "keypoints": {0: "batch2", 1: "num_keypoints"},
            "descriptors": {0: "batch2", 1: "num_keypoints"},
            "matches": {0: "num_matches"},
            "mscores": {0: "num_matches"},
        },
        opset_version=17,
    )
    print(f"Exported to {out_path}")


if __name__ == "__main__":
    main()
