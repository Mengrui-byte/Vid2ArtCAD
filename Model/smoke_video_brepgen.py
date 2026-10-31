"""Check conditioning and gradients on an unmodified official BrepGen network."""

import torch

from Model.video_brep_condition import VideoConditionedBrepGen
from network import EdgePosNet, EdgeZNet, SurfPosNet, SurfZNet


def main():
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = VideoConditionedBrepGen({
        "surfpos": SurfPosNet(False),
        "surfz": SurfZNet(False),
        "edgepos": EdgePosNet(False),
        "edgez": EdgeZNet(False),
    }).to(device)
    # Only surfpos is evaluated here; the other three models use distinct input signatures.
    noisy = torch.randn(1, 4, 6, device=device)
    time = torch.tensor([100], device=device)
    video = torch.randn(1, 8, 192, device=device, requires_grad=True)
    prediction = adapter("surfpos", video, noisy, time, None)
    prediction.square().mean().backward()
    assert prediction.shape == noisy.shape
    assert video.grad is not None and video.grad.abs().sum().item() > 0
    print(f"official_surfpos_shape={tuple(prediction.shape)} video_gradient={video.grad.abs().sum().item():.6g}")


if __name__ == "__main__":
    main()
