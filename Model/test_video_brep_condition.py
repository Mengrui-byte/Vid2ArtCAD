"""Regression checks for video conditioning of the BrepGen denoisers."""

import unittest

import torch
from torch import nn

from Model.video_brep_condition import VideoConditionedBrepGen


class DummyDenoiser(nn.Module):
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.time_embed = nn.Linear(8, 8)
        self.output = nn.Linear(8, 6)

    def forward(self, x, time, class_label=None):
        return self.output(x + self.time_embed(time).unsqueeze(1))


class VideoConditionTest(unittest.TestCase):
    def test_video_changes_prediction_and_receives_gradient(self):
        torch.manual_seed(7)
        adapter = VideoConditionedBrepGen(
            {name: DummyDenoiser() for name in VideoConditionedBrepGen.BRANCHES},
            video_dim=4,
        )
        noisy = torch.randn(2, 3, 8)
        timestep = torch.randn(2, 8)
        video = torch.randn(2, 5, 4, requires_grad=True)
        for branch in VideoConditionedBrepGen.BRANCHES:
            prediction = adapter(branch, video, noisy, timestep)
            changed = adapter(branch, video * torch.tensor([1.0, 2.0, 0.5, -1.0]), noisy, timestep)
            self.assertFalse(torch.allclose(prediction, changed))
            prediction.square().mean().backward(retain_graph=True)
            self.assertGreater(adapter.video_projection[branch][1].weight.grad.abs().sum(), 0)
        self.assertGreater(video.grad.abs().sum(), 0)


if __name__ == "__main__":
    unittest.main()
