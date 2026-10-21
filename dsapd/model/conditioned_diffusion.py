# Modified: variable names aligned with the method diagram.
from typing import Tuple, Set, List, Dict

import torch
from torch import nn

from .prior_controlnet import ControlledUnetModel, ControlNet
from .vae import AutoencoderKL
from .util import GroupNorm32
from .clip import FrozenOpenCLIPEmbedder
from .distributions import DiagonalGaussianDistribution
from ..utils.tilevae import VAEHook
from utils.analytic_priors import extract_analytic_priors


def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class ControlLDM(nn.Module):

    def __init__(
        self, unet_cfg, vae_cfg, clip_cfg, controlnet_cfg, latent_scale_factor
    ):
        super().__init__()
        if "prior_hint_channels" in unet_cfg:
            prior_hint_channels = unet_cfg.pop("prior_hint_channels")
            controlnet_cfg.setdefault("prior_hint_channels", prior_hint_channels)
            print(
                "[ControlLDM] moved prior_hint_channels from unet_cfg to "
                "controlnet_cfg. Please keep it under controlnet_cfg in yaml."
            )
        self.unet = ControlledUnetModel(**unet_cfg)
        self.vae = AutoencoderKL(**vae_cfg)
        self.clip = FrozenOpenCLIPEmbedder(**clip_cfg)
        self.controlnet = ControlNet(**controlnet_cfg)
        self.scale_factor = latent_scale_factor
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def load_pretrained_sd(
        self, sd: Dict[str, torch.Tensor]
    ) -> Tuple[Set[str], Set[str]]:
        module_map = {
            "unet": "model.diffusion_model",
            "vae": "first_stage_model",
            "clip": "cond_stage_model",
        }
        modules = [("unet", self.unet), ("vae", self.vae), ("clip", self.clip)]
        used = set()
        missing = set()
        for name, module in modules:
            init_sd = {}
            scratch_sd = module.state_dict()
            for key in scratch_sd:
                target_key = ".".join([module_map[name], key])
                if target_key not in sd:
                    missing.add(target_key)
                    continue
                init_sd[key] = sd[target_key].clone()
                used.add(target_key)
            module.load_state_dict(init_sd, strict=False)
        unused = set(sd.keys()) - used
        for module in [self.vae, self.clip, self.unet]:
            module.eval()
            module.train = disabled_train
            for p in module.parameters():
                p.requires_grad = False
        return unused, missing

    @torch.no_grad()
    def load_controlnet_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:
        if isinstance(sd, dict) and "controlnet" in sd:
            sd = sd["controlnet"]
        current = self.controlnet.state_dict()
        has_prior_branch = any(k.startswith("prior_") for k in current)
        if has_prior_branch:
            filtered = {k: v for k, v in sd.items() if k in current and current[k].shape == v.shape}
            missing, unexpected = self.controlnet.load_state_dict(filtered, strict=False)
            missing_prior = [k for k in missing if k.startswith("prior_")]
            missing_other = [k for k in missing if not k.startswith("prior_")]
            if missing_other:
                raise RuntimeError(f"missing non-prior ControlNet keys: {missing_other}")
            if missing_prior:
                print(f"[ControlLDM] initialize new prior branch keys from scratch: {missing_prior}")
            if unexpected:
                print(f"[ControlLDM] unexpected ControlNet keys ignored: {unexpected}")
        else:
            self.controlnet.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def load_controlnet_from_unet(self) -> Tuple[Set[str]]:
        unet_sd = self.unet.state_dict()
        scratch_sd = self.controlnet.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()
        for key in scratch_sd:
            if key in unet_sd:
                this, target = scratch_sd[key], unet_sd[key]
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        self.controlnet.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch

    def vae_encode(
        self,
        image: torch.Tensor,
        sample: bool = True,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        if tiled:
            def encoder(x: torch.Tensor) -> DiagonalGaussianDistribution:
                h = VAEHook(
                    self.vae.encoder,
                    tile_size=tile_size,
                    is_decoder=False,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(x)
                moments = self.vae.quant_conv(h)
                posterior = DiagonalGaussianDistribution(moments)
                return posterior
        else:
            encoder = self.vae.encode

        if sample:
            z = encoder(image).sample() * self.scale_factor
        else:
            z = encoder(image).mode() * self.scale_factor
        return z

    def vae_decode(
        self,
        z: torch.Tensor,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        if tiled:
            def decoder(z):
                z = self.vae.post_quant_conv(z)
                dec = VAEHook(
                    self.vae.decoder,
                    tile_size=tile_size,
                    is_decoder=True,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(z)
                return dec
        else:
            decoder = self.vae.decode
        return decoder(z / self.scale_factor)

    def prepare_condition(
        self,
        cond_img: torch.Tensor,
        txt: List[str],
        tiled: bool = False,
        tile_size: int = -1,
    ) -> Dict[str, torch.Tensor]:
        c_img = self.vae_encode(
            cond_img * 2 - 1,
            sample=False,
            tiled=tiled,
            tile_size=tile_size,
        )
        cond = dict(
            c_txt=self.clip.encode(txt),
            c_img=c_img,
        )
        prior_channels = getattr(self.controlnet, "prior_hint_channels", 0)
        if prior_channels > 0:
            P_I = extract_analytic_priors(cond_img, out_size=c_img.shape[-2:])
            if P_I.shape[1] > prior_channels:
                P_I = P_I[:, :prior_channels]
            elif P_I.shape[1] < prior_channels:
                pad = torch.zeros(
                    P_I.shape[0],
                    prior_channels - P_I.shape[1],
                    P_I.shape[2],
                    P_I.shape[3],
                    device=P_I.device,
                    dtype=P_I.dtype,
                )
                P_I = torch.cat([P_I, pad], dim=1)
            cond["c_prior"] = P_I
        return cond

    def prepare_text_condition(
        self,
        txt: List[str],
    ) -> Dict[str, torch.Tensor]:
        return dict(
            c_txt=self.clip.encode(txt),
            c_img=None,
        )

    def forward(self, z_t, t, cond):
        c_txt = cond["c_txt"]
        c_img = cond["c_img"]
        P_I = cond.get("c_prior", None)
        control = self.controlnet(
            x=z_t,
            hint=c_img,
            timesteps=t,
            context=c_txt,
            prior_hint=P_I,
        )
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        eps = self.unet(
            x=z_t,
            timesteps=t,
            context=c_txt,
            control=control,
            only_mid_control=False,
        )
        return eps

    def cast_dtype(self, dtype: torch.dtype) -> "ControlLDM":
        self.unet.dtype = dtype
        self.controlnet.dtype = dtype
        # convert unet blocks to dtype
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.type(dtype)
        # convert controlnet blocks and zero-convs to dtype
        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.type(dtype)
        if getattr(self.controlnet, "prior_projs", None) is not None:
            self.controlnet.prior_projs.type(dtype)
            self.controlnet.prior_middle_proj.type(dtype)

        def cast_groupnorm_32(m):
            if isinstance(m, GroupNorm32):
                m.type(torch.float32)

        # GroupNorm32 only works with float32
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.apply(cast_groupnorm_32)
        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.apply(cast_groupnorm_32)
        if getattr(self.controlnet, "prior_projs", None) is not None:
            self.controlnet.prior_projs.apply(cast_groupnorm_32)
            self.controlnet.prior_middle_proj.apply(cast_groupnorm_32)
