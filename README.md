
# DSAPD: Dual-End Spatial Conditioning and Analytic-Prior-Guided Diffusion for Real-World Dehazing


## Visualization

### Spatially Varying Haze

[View Spatially Varying Haze](hazeregion_final.pdf)

### Visual Comparison on RTTS

[View RTTS Comparison](RTTS_final.pdf)

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
```

这样 GitHub 项目主页会依次显示标题、两张效果图、环境配置和数据集下载链接。

如果你还想让别人点击查看原 PDF，可以分别在两张图片下面添加：

```markdown
[View Haze Region PDF](hazeregion_final.pdf)

[View RTTS Comparison PDF](RTTS_final%281%29.pdf)
```

这里假设 PDF 和 README 在同一目录，且 RTTS 文件名确实带 `(1)`。若你在 GitHub 上已将其重命名为 `RTTS_final.pdf`，就把链接改成对应文件名。

**我建议采用现在这种布局：标题下面先展示两张图，再放环境和数据集。** 对于图像恢复论文的开源仓库，进入主页就能看到直观的恢复效果，比仅提供 PDF 链接更合适。
