"""Video-conditioned structured B-rep latent and kinematics models."""

__all__ = ["VideoJEPAAssemblyModel"]


def __getattr__(name):
    if name == "VideoJEPAAssemblyModel":
        from .model import VideoJEPAAssemblyModel
        return VideoJEPAAssemblyModel
    raise AttributeError(name)
