"""Dual-view pretrained V-JEPA 2.1 encoder with OBJ and URDF heads."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class VideoJEPAEncoder(nn.Module):
    """Official Meta V-JEPA 2.1 ViT-B encoder loaded from a local checkpoint."""

    def __init__(
        self,
        dim: int = 192,
        repo_dir: str = "/root/vjepa2",
        checkpoint: str = "/root/.cache/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
        num_frames: int = 16,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        backbone_and_predictor = torch.hub.load(
            repo_dir,
            "vjepa2_1_vit_base_384",
            source="local",
            pretrained=False,
            num_frames=num_frames,
        )
        self.backbone = backbone_and_predictor[0]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get(
            "ema_encoder",
            payload.get("target_encoder", payload.get("encoder", payload.get("model", payload))),
        )
        state = {self._strip_prefix(key): value for key, value in state.items()}
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        loaded = len(state) - len(unexpected)
        if loaded < 100:
            raise RuntimeError(
                f"V-JEPA checkpoint did not match the official ViT-B backbone: "
                f"loaded={loaded}, missing={len(missing)}, unexpected={len(unexpected)}"
            )
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        hidden_size = getattr(self.backbone, "embed_dim", 768)
        self.projection = nn.Linear(hidden_size, dim)
        self.checkpoint = checkpoint

    @staticmethod
    def _strip_prefix(key: str) -> str:
        for prefix in ("module.", "backbone.", "encoder."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        return key

    def forward(self, pixel_values_videos: torch.Tensor, mask_ratio: float | None = None):
        if self.freeze_backbone:
            with torch.no_grad():
                raw_features = self.backbone(pixel_values_videos)
        else:
            raw_features = self.backbone(pixel_values_videos)
        if isinstance(raw_features, (tuple, list)):
            raw_features = raw_features[0]
        features = self.projection(raw_features)
        return features, features.new_zeros(())

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum: float = 0.996) -> None:
        return None


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000.0) / max(half - 1, 1)))
        phase = t.float().unsqueeze(-1) * freq.unsqueeze(0)
        emb = torch.cat((phase.sin(), phase.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.net(emb)


class BrepGenStructuredDiffusion(nn.Module):
    """Top-down structured node diffusion: solid/root, faces, edges, vertices.

    Continuous features contain a normalized node bbox plus a local shape code
    from the BrepGen surface/edge VAE. Discrete node type and parent predictions
    represent the hierarchical incidence tree; repeated geometric nodes are
    merged by the topology postprocessor after decoding.
    """

    NODE_TYPES = ("solid", "face", "edge", "vertex", "pad")

    def __init__(self, dim: int, max_nodes: int, shape_dim: int = 48, layers: int = 4, heads: int = 4):
        super().__init__()
        self.max_nodes = max_nodes
        self.shape_dim = shape_dim
        self.feature_dim = 6 + shape_dim
        self.node_position = nn.Parameter(torch.randn(1, max_nodes, dim) * 0.02)
        self.input = nn.Linear(self.feature_dim, dim)
        self.time = TimeEmbedding(dim)
        layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, batch_first=True, norm_first=True)
        self.denoiser = nn.TransformerDecoder(layer, layers)
        self.output = nn.Linear(dim, self.feature_dim)
        self.node_type = nn.Linear(dim, len(self.NODE_TYPES))
        self.parent = nn.Linear(dim, max_nodes + 1)
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, 1000))
        alphas = 1.0 - self.betas
        self.register_buffer("alpha_bar", torch.cumprod(alphas, dim=0))

    def predict_noise(self, noisy_nodes: torch.Tensor, timestep: torch.Tensor, video_tokens: torch.Tensor):
        x = self.input(noisy_nodes) + self.node_position
        x = x + self.time(timestep).unsqueeze(1)
        hidden = self.denoiser(x, video_tokens)
        return self.output(hidden), self.node_type(hidden), self.parent(hidden)

    def diffusion_loss(self, clean_nodes: torch.Tensor, node_mask: torch.Tensor, video_tokens: torch.Tensor):
        b = clean_nodes.shape[0]
        t = torch.randint(0, self.betas.numel(), (b,), device=clean_nodes.device)
        eps = torch.randn_like(clean_nodes)
        a = self.alpha_bar[t].view(b, 1, 1)
        noisy = a.sqrt() * clean_nodes + (1 - a).sqrt() * eps
        eps_hat, type_logits, parent_logits = self.predict_noise(noisy, t, video_tokens)
        weights = node_mask.unsqueeze(-1).float()
        noise_loss = ((eps_hat - eps).square() * weights).sum() / weights.sum().clamp_min(1)
        return {"noise": noise_loss, "type_logits": type_logits, "parent_logits": parent_logits}

    @torch.no_grad()
    def sample(self, video_tokens: torch.Tensor, steps: int = 100, node_count: int | None = None):
        b = video_tokens.shape[0]
        n = node_count or self.max_nodes
        x = torch.randn(b, n, self.feature_dim, device=video_tokens.device)
        schedule = torch.linspace(self.betas.numel() - 1, 0, steps, device=x.device).long()
        types = parents = None
        for i, ti in enumerate(schedule):
            t = torch.full((b,), int(ti), device=x.device, dtype=torch.long)
            eps, types, parents = self.predict_noise(x, t, video_tokens)
            abar = self.alpha_bar[ti]
            clean = (x - (1 - abar).sqrt() * eps) / abar.sqrt()
            if i + 1 < len(schedule):
                abar_prev = self.alpha_bar[schedule[i + 1]]
                x = abar_prev.sqrt() * clean + (1 - abar_prev).sqrt() * eps
            else:
                x = clean
        return {"node_features": x, "node_type_logits": types, "parent_logits": parents}


class KinematicsBranch(nn.Module):
    """Independent part-slot decoder for the assembly kinematic tree."""

    def __init__(self, dim: int, max_parts: int):
        super().__init__()
        self.max_parts = max_parts
        self.queries = nn.Parameter(torch.randn(1, max_parts, dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.present = nn.Linear(dim, 1)
        self.parent = nn.Linear(dim, max_parts + 1)
        self.joint_type = nn.Linear(dim, 3)
        self.axis = nn.Linear(dim, 3)
        self.origin = nn.Linear(dim, 3)
        self.rotation6d = nn.Linear(dim, 6)
        nn.init.zeros_(self.rotation6d.weight)
        with torch.no_grad():
            self.rotation6d.bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))
        self.limits = nn.Linear(dim, 2)

    def forward(self, video_tokens: torch.Tensor):
        q = self.queries.expand(video_tokens.shape[0], -1, -1)
        slots, _ = self.cross_attention(q, video_tokens, video_tokens, need_weights=False)
        slots = self.norm(slots + q)
        return {
            "slots": slots,
            "present": self.present(slots).squeeze(-1),
            "parent": self.parent(slots),
            "joint_type": self.joint_type(slots),
            "axis": self.axis(slots),
            "origin": self.origin(slots),
            "rotation6d": self.rotation6d(slots),
            "limits": self.limits(slots),
        }


class OBJVertexDecoder(nn.Module):
    """Fixed-topology mesh decoder for the single-assembly overfit experiment."""

    def __init__(self, dim: int, max_parts: int, max_vertices: int):
        super().__init__()
        self.max_parts = max_parts
        self.max_vertices = max_vertices
        self.vertex_queries = nn.Parameter(torch.randn(1, 1, max_vertices, dim) * 0.02)
        self.decoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
            nn.Tanh(),
        )

    def forward(self, part_slots: torch.Tensor) -> torch.Tensor:
        queries = part_slots.unsqueeze(2) + self.vertex_queries
        return self.decoder(queries)


class DualViewFusion(nn.Module):
    def __init__(self, dim: int, layers: int = 2, tokens_per_view: int = 128):
        super().__init__()
        self.tokens_per_view = tokens_per_view
        self.view_embedding = nn.Parameter(torch.randn(2, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)

    def forward(self, tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> torch.Tensor:
        tokens_a = F.adaptive_avg_pool1d(tokens_a.transpose(1, 2), self.tokens_per_view).transpose(1, 2)
        tokens_b = F.adaptive_avg_pool1d(tokens_b.transpose(1, 2), self.tokens_per_view).transpose(1, 2)
        a = tokens_a + self.view_embedding[0]
        b = tokens_b + self.view_embedding[1]
        return self.encoder(torch.cat((a, b), dim=1))


class VideoJEPAAssemblyModel(nn.Module):
    def __init__(self, max_parts: int = 16, max_brep_nodes: int = 256, dim: int = 192, shape_dim: int = 48, max_vertices: int = 4096, vjepa_repo: str = "/root/vjepa2", vjepa_checkpoint: str = "/root/.cache/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt", vjepa_num_frames: int = 16):
        super().__init__()
        self.encoder_a = VideoJEPAEncoder(dim=dim, repo_dir=vjepa_repo, checkpoint=vjepa_checkpoint, num_frames=vjepa_num_frames)
        self.encoder_b = VideoJEPAEncoder(dim=dim, repo_dir=vjepa_repo, checkpoint=vjepa_checkpoint, num_frames=vjepa_num_frames)
        self.fusion = DualViewFusion(dim)
        self.brep = BrepGenStructuredDiffusion(dim, max_brep_nodes, shape_dim)
        self.kinematics = KinematicsBranch(dim, max_parts)
        self.mesh = OBJVertexDecoder(dim, max_parts, max_vertices)

    def forward_from_features(self, tokens_a: torch.Tensor, tokens_b: torch.Tensor, brep_nodes: torch.Tensor | None = None, brep_mask: torch.Tensor | None = None):
        video_tokens = self.fusion(tokens_a, tokens_b)
        jepa_loss = video_tokens.new_zeros(())
        kinematics = self.kinematics(video_tokens)
        mesh_vertices = self.mesh(kinematics["slots"])
        brep_loss = None
        if brep_nodes is not None:
            if brep_mask is None:
                raise ValueError("brep_mask is required with brep_nodes")
            brep_loss = self.brep.diffusion_loss(brep_nodes, brep_mask, video_tokens)
        return {"kinematics": kinematics, "mesh_vertices": mesh_vertices, "brep_diffusion": brep_loss}, jepa_loss

    def forward(self, video_a: torch.Tensor, video_b: torch.Tensor, brep_nodes: torch.Tensor | None = None, brep_mask: torch.Tensor | None = None):
        tokens_a, _ = self.encoder_a(video_a)
        tokens_b, _ = self.encoder_b(video_b)
        return self.forward_from_features(tokens_a, tokens_b, brep_nodes, brep_mask)
