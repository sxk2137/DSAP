# Modified: variable names aligned with the method diagram.
import torch
import torch.nn.functional as F


def _to_01(img: torch.Tensor) -> torch.Tensor:
    if img.min().detach() < -0.1:
        img = (img + 1.0) * 0.5
    return img.clamp(0.0, 1.0)


def _min_filter(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    pad = patch_size // 2
    return -F.max_pool2d(-x, kernel_size=patch_size, stride=1, padding=pad)


def _max_filter(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    pad = patch_size // 2
    return F.max_pool2d(x, kernel_size=patch_size, stride=1, padding=pad)


def _dark_channel(img: torch.Tensor, patch_size: int) -> torch.Tensor:
    return _min_filter(img.min(dim=1, keepdim=True).values, patch_size)


def _bright_channel(img: torch.Tensor, patch_size: int) -> torch.Tensor:
    return _max_filter(img.max(dim=1, keepdim=True).values, patch_size)


def _atmospheric_light(img: torch.Tensor, dark: torch.Tensor, topk_ratio: float) -> torch.Tensor:
    b, c, h, w = img.shape
    k = max(1, int(h * w * topk_ratio))
    _, idx = torch.topk(dark.reshape(b, -1), k=k, dim=1, largest=True, sorted=False)
    img_flat = img.reshape(b, c, -1)
    idx = idx.unsqueeze(1).expand(-1, c, -1)
    candidates = torch.gather(img_flat, 2, idx)
    best = candidates.mean(dim=1).argmax(dim=1, keepdim=True)
    best = best.unsqueeze(1).expand(-1, c, -1)
    return candidates.gather(2, best).view(b, c, 1, 1).clamp(0.05, 1.0)


def _edge_map(img: torch.Tensor) -> torch.Tensor:
    gray = img.mean(dim=1, keepdim=True)
    dx = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
    dy = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    edge = dx + dy
    return edge / edge.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)


@torch.no_grad()
def extract_analytic_priors(
    I: torch.Tensor,
    out_size=None,
    patch_size: int = 15,
    omega: float = 0.95,
    topk_ratio: float = 0.001,
) -> torch.Tensor:
    """Return [dark, bright, transmission, density, edge] in [0, 1]."""
    if patch_size % 2 == 0:
        patch_size += 1

    img = _to_01(I.float())
    dark = _dark_channel(img, patch_size)
    bright = _bright_channel(img, patch_size)
    A_rgb = _atmospheric_light(img, dark, topk_ratio)
    transmission = 1.0 - omega * _dark_channel(img / (A_rgb + 1e-6), patch_size)
    transmission = transmission.clamp(0.05, 1.0)
    density = (1.0 - transmission).clamp(0.0, 1.0)
    edge = _edge_map(img).clamp(0.0, 1.0)

    P_I = torch.cat(
        [dark.clamp(0.0, 1.0), bright.clamp(0.0, 1.0), transmission, density, edge],
        dim=1,
    )
    if out_size is not None and P_I.shape[-2:] != tuple(out_size):
        P_I = F.interpolate(P_I, size=out_size, mode="bilinear", align_corners=False)
    return P_I.to(device=I.device, dtype=I.dtype).contiguous()
