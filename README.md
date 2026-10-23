
# DSAPD: Dual-End Spatial Conditioning and Analytic-Prior-Guided Diffusion for Real-World Dehazing


## Visualization

### Spatially Varying Haze

![Spatially Varying Haze](https://raw.githubusercontent.com/sxk2137/DSAP/main/hazeregion_final.png)

### Visual Comparison on RTTS

![RTTS Comparison](https://raw.githubusercontent.com/sxk2137/DSAP/main/RTTS_final.png)

## Environment Setup

```bash
git clone https://github.com/sxk2137/DSAP.git
cd DSAP/DSAP

conda create -n h2h python=3.10 -y
conda activate h2h

pip install -r requirements.txt
```

## Datasets

| Dataset | Download Link |
|---------|---------------|
| RIDCP | [Download](https://github.com/RQ-Wu/RIDCP_dehazing) |
| OTS | [Download](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2) |
| URHI | [Download](https://pan.baidu.com/s/1i5GiZsx) (Code: vt99) |
| RTTS | [Download](https://pan.baidu.com/s/1A0MMAnlWmuJ0dXhsbXk4Gg) (Code: 4mv7) |
| Fattal | [Download](https://www.cs.huji.ac.il/w~raananf/projects/dehaze_cl/results/) |

