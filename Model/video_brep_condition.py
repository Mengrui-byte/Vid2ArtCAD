"""Video-conditioned adapter for the four official BrepGen denoisers.

This module does not replace BrepGen's VAE or STEP topology reconstruction.
It injects learned video conditioning into the time embedding used by each
official denoiser, preserving its original state-dict keys and output shapes.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn


class VideoConditionedBrepGen(nn.Module):
    BRANCHES = ("surfpos", "surfz", "edgepos", "edgez")

    def __init__(self, denoisers: dict[str, nn.Module], video_dim: int = 192,
                 required_branches: tuple[str, ...] | None = None):
        super().__init__()
        missing = set(required_branches or self.BRANCHES) - set(denoisers)
        if missing:
            raise ValueError(f"missing BrepGen denoisers: {sorted(missing)}")
        unknown = set(denoisers) - set(self.BRANCHES)
        if unknown:
            raise ValueError(f"unknown BrepGen denoisers: {sorted(unknown)}")
        self.denoisers = nn.ModuleDict(denoisers)
        self.video_projection = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(video_dim), nn.Linear(video_dim, net.embed_dim))
            for name, net in self.denoisers.items()
        })
        self.condition_attention = nn.ModuleDict({
            name: nn.MultiheadAttention(net.embed_dim, 8, batch_first=True)
            for name, net in self.denoisers.items()
        })
        self.condition_gate = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(-2.0)) for name in self.denoisers
        })
        self._active_condition: torch.Tensor | None = None
        self._active_tokens: torch.Tensor | None = None
        self._hooks = [
            net.time_embed.register_forward_hook(self._condition_hook(name))
            for name, net in self.denoisers.items()
        ]
        self._hooks.extend(
            net.net.register_forward_pre_hook(self._attention_hook(name), with_kwargs=True)
            for name, net in self.denoisers.items() if hasattr(net, "net")
        )

    def _condition_hook(self, name: str):
        def add_condition(_module, _inputs, output):
            if self._active_condition is None:
                return output
            if output.shape[0] != self._active_condition.shape[0]:
                raise ValueError("video tokens and BrepGen batch size differ")
            return output + self.video_projection[name](self._active_condition)
        return add_condition

    def _attention_hook(self, name: str):
        def add_cross_attention(module, args, kwargs):
            if self._active_tokens is None:
                return args, kwargs
            src = kwargs.get("src", args[0] if args else None)
            if src is None or src.shape[1] != self._active_tokens.shape[0]:
                raise ValueError("BrepGen transformer and video batch size differ")
            context = self.video_projection[name](self._active_tokens)
            query = src.transpose(0, 1)
            attended, _ = self.condition_attention[name](query, context, context, need_weights=False)
            modified = (query + self.condition_gate[name].sigmoid() * attended).transpose(0, 1)
            if "src" in kwargs:
                return args, {**kwargs, "src": modified}
            return (modified, *args[1:]), kwargs
        return add_cross_attention

    @contextmanager
    def conditioned(self, video_tokens: torch.Tensor):
        if self._active_condition is not None:
            raise RuntimeError("nested video conditioning is not supported")
        if video_tokens.ndim != 3:
            raise ValueError("expected video tokens [batch, sequence, dim]")
        self._active_condition = video_tokens.mean(dim=1)
        self._active_tokens = video_tokens
        try:
            yield
        finally:
            self._active_condition = None
            self._active_tokens = None

    def forward(self, branch: str, video_tokens: torch.Tensor, *args, **kwargs):
        if branch not in self.denoisers:
            raise ValueError(f"unknown BrepGen branch: {branch}")
        with self.conditioned(video_tokens):
            return self.denoisers[branch](*args, **kwargs)

    def load_brepgen_weights(self, paths: dict[str, str]) -> None:
        for name in self.denoisers:
            state = torch.load(paths[name], map_location="cpu", weights_only=False)
            self.denoisers[name].load_state_dict(state, strict=True)


def build_brepgen_adapter(video_dim: int = 192) -> VideoConditionedBrepGen:
    from network import EdgePosNet, EdgeZNet, SurfPosNet, SurfZNet

    return VideoConditionedBrepGen({
        "surfpos": SurfPosNet(False),
        "surfz": SurfZNet(False),
        "edgepos": EdgePosNet(False),
        "edgez": EdgeZNet(False),
    }, video_dim=video_dim)


class VideoBrepAssemblyModel(nn.Module):
    """Dual-view V-JEPA features shared by BrepGen and the URDF decoder."""

    def __init__(self, brepgen: VideoConditionedBrepGen, max_parts: int = 16, dim: int = 192):
        super().__init__()
        from .model import DualViewFusion, KinematicsBranch

        self.fusion = DualViewFusion(dim)
        self.brepgen = brepgen
        self.kinematics = KinematicsBranch(dim, max_parts)

    def forward_from_features(
        self, tokens_a: torch.Tensor, tokens_b: torch.Tensor,
        branch: str, *brep_inputs, **brep_kwargs,
    ):
        video_tokens = self.fusion(tokens_a, tokens_b)
        return {
            "brep_noise": self.brepgen(branch, video_tokens, *brep_inputs, **brep_kwargs),
            "kinematics": self.kinematics(video_tokens),
        }
