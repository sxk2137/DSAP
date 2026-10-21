# DSAPD

**Dual-End Spatial Conditioning and Analytic-Prior-Guided Diffusion for Real-World Dehazing**

Implementation of spatial haze synthesis, paired–unpaired data handling, and
analytic prior injection into a diffusion-based dehazing architecture.

## Architecture

**Stage I: Spatially Varying Haze Augmentation.** A stochastic spatial field
modulates the scattering coefficient. Depth-aware transmission and atmospheric
light produce a synthetic hazy image. `HybridTrainingData` combines synthetic
pairs with real hazy images using the mixture probability `p`.

**Stage II: Analytic-Prior-Guided Diffusion Dehazing.** The input image provides
both a VAE image condition and a five-channel analytic prior. The prior is
aligned to each ControlNet feature scale, projected by a 3 × 3 convolution,
and added through a sigmoid gate. Encoder and middle-block features provide
zero-convolution residuals to the diffusion U-Net.

## Project structure

```text
configs/
  train/                Stage I and Stage II model and data configurations
  inference/            Model and sampling configurations
datasets.py             Real, paired, and paired–unpaired dataset definitions
dsapd/
  model/
    conditioned_diffusion.py  Latent diffusion model and condition construction
    prior_controlnet.py       ControlNet, prior injection, and controlled U-Net
    unet.py                   U-Net architecture
    vae.py                    Variational autoencoder
    gaussian_diffusion.py
    attention.py
    clip.py
    open_clip/          Text encoder and tokenizer vocabulary
    ...
  sampler/              Diffusion sampling components
  utils/                Tiled VAE and image utilities
  image_ops.py           Image resizing and padding utilities
utils/
  spatial_haze.py       Stochastic spatial haze synthesis
  analytic_priors.py    Five-channel analytic prior extraction
  data_utils.py         Image degradation and data operations
```

## Notation

| Figure | Implementation | Meaning |
| --- | --- | --- |
| $J$ | `J` | Clean image |
| $d(x)$ | `d` | Input depth map |
| $\widetilde\beta(x)$ | `beta_tilde` | Stochastic spatial field |
| $\beta,\alpha$ | `beta`, `alpha` | Base scattering and spatial modulation strength |
| $\beta_s(x)$ | `beta_s` | Spatial scattering coefficient |
| $t_s(x)$ | `t_s` | Transmission |
| $A_{\mathrm{rgb}}$ | `A_rgb` | Atmospheric light |
| $I_s,I_r$ | `I_s`, `I_r` | Synthetic and real hazy images |
| $I$ | `I` | Hazy input to prior extraction |
| $P(I)$ | `P_I` | Analytic prior stack |
| $z_t$ | `z_t` | Noisy latent |
| $c_{\mathrm{img}}$ | `c_img` | VAE image condition |
| $F_l$ | `F_l` | Feature at an injection site |
| $g_l$ | `g_l` | Gate logit at an injection site |

Layer attributes and parameter keys retain their original names, including
`prior_projs`, `prior_gates`, `prior_middle_proj`, and `prior_middle_gate`.

## Spatial haze synthesis

`utils.spatial_haze.synthesize_spatial_haze(J, d, beta_tilde=...)` computes

```text
beta_s = beta + alpha * beta_tilde
t_s = exp(-2.0 * d_eff * beta_s)
I_s = J_degraded * t_s + A_rgb * (1 - t_s)
```

The depth convention is `d_eff = 1 - normalize(d)`. Normalization uses `1e-8`
in the denominator. `J` is a BGR floating-point image in [0, 1]; the returned
`I_s` is an RGB uint8 image. Atmospheric light follows the BGR working channel
order before the final RGB conversion. Gamma and Gaussian degradation precede
haze composition, and JPEG degradation follows it.

`generate_spatial_field` constructs a normalized spatial field, while
`generate_spatial_field_bank` writes a collection of fields. Both Python's `random` and
NumPy contribute to random sampling.

## Analytic priors and injection

`utils.analytic_priors.extract_analytic_priors(I)` returns a B × 5 × H × W tensor in the order

```text
[dark, bright, transmission, density, edge]
```

The input is an RGB floating-point tensor. If its minimum is below -0.1, it is
mapped from [-1, 1] to [0, 1]; otherwise it is clamped directly. Prior extraction
does not require gradients. The returned stack retains the input dtype and
device, with optional bilinear resizing through `out_size`.

At each ControlNet injection site,

```text
P_l = Conv3x3_l(bilinear_resize(P_I, F_l.shape[-2:]))
F_hat_l = F_l + sigmoid(g_l) * P_l
```

Each projection and gate is independent. Gate logits are initialized to -4.0.
The prior channel count is configured through `prior_hint_channels`.

## Environment and configuration

The dependency file specifies PyTorch 2.2.2 with CUDA 11.8 and the accompanying
libraries. Use Python 3.10 and install from the project directory:

```bash
python -m pip install -r requirements.txt
```

Configuration paths are relative to the project directory. The model sections
define the VAE, text encoder, diffusion U-Net, ControlNet, and diffusion schedule.
The data sections define paired and real-image sources.

## Acknowledgments

This project builds on
[Learning Hazing to Dehazing](https://github.com/ruiyi-w/Learning-Hazing-to-Dehazing)
and [DiffBIR](https://github.com/XPixelGroup/DiffBIR). The stochastic spatial
field follows the MCBM formulation used in HazeFlow.

See [NOTICE](NOTICE) for attribution and [LICENSE](LICENSE) for the retained
Apache License, Version 2.0.
