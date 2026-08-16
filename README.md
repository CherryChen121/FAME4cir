<div align="center">

# FAME-ViL: Multi-Tasking Vision-Language Model for Heterogeneous Fashion Tasks

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://mmf.sh/"><img alt="MMF" src="https://img.shields.io/badge/MMF-0054a6?logo=meta&logoColor=white"></a>
[![Conference](http://img.shields.io/badge/CVPR-2023(Highlight)-6790AC.svg)](https://cvpr.thecvf.com/)
[![Paper](http://img.shields.io/badge/Paper-arxiv.2303.02483-B31B1B.svg)](https://arxiv.org/abs/2303.02483)

</div>

## Updates
- :heart_eyes: (21/03/2023) Our FAME-ViL is selected as a **highlight paper** at CVPR 2023! (**Top 2.5%** of 9155 submissions)
- :blush: (12/03/2023) Code released!

Our trained model is available at [Google Drive](https://drive.google.com/drive/folders/17YflGKqt4sLbsfCSKZTGzdwaP9JcO7aN?usp=sharing).

Please refer to [FashionViL repo](https://github.com/BrandonHanx/mmf#data-preparation) for the dataset preparation.

Test on FashionIQ
```
python mmf_cli/run.py \
config=projects/fashionclip/configs/mtl_wa.yaml \
model=fashionclip \
datasets=fashioniq \
checkpoint.resume_file=save/backup_ckpts/fashionclip_512.pth \
run_type=test \
model_config.fashionclip.adapter_config.bottleneck=512
```

## Combined Fundus CIR

The repository also supports the FashionIQ-style dataset at
`/data0/qrchen/datasets/Combined_Fundus_CIR_Dataset`.

- Training and validation use `Internal.train` and `Internal.val`.
- Testing can use `Internal.test`, `GRAPE.test`, or `ODIR5K.test`.
- Evaluation ranks every query against the complete gallery from
  `image_splits/split.<subset>.<split>.json`.

Run from the repository root:

```bash
# CLIP ViT-B/16 (batch size 32)
CUDA_VISIBLE_DEVICES=3 bash 命令.sh train_combined

# CLIP ViT-L/14 (batch size 16)
CUDA_VISIBLE_DEVICES=3 bash 命令.sh train_combined_L
```

After training, evaluate the three test sets with the matching backbone:

```bash
# CLIP ViT-B/16
CUDA_VISIBLE_DEVICES=3 bash 命令.sh test_combined_internal
CUDA_VISIBLE_DEVICES=3 bash 命令.sh test_combined_grape
CUDA_VISIBLE_DEVICES=3 bash 命令.sh test_combined_odir

# CLIP ViT-L/14
CUDA_VISIBLE_DEVICES=3 bash 命令.sh test_combined_internal_L
CUDA_VISIBLE_DEVICES=3 bash 命令.sh test_combined_grape_L
CUDA_VISIBLE_DEVICES=3 bash 命令.sh test_combined_odir_L
```

The ViT-B/16 test commands use
`save/fashionclip_combined_fundus_composition_xattn/fashionclip_final.pth`
by default. The ViT-L/14 commands use
`save/fashionclip_combined_fundus_vitL14_composition_xattn/fashionclip_final.pth`.
To evaluate another checkpoint:

```bash
# CLIP ViT-B/16
FAME_CHECKPOINT=/absolute/path/to/fashionclip_final.pth \
CUDA_VISIBLE_DEVICES=3 bash 命令.sh test_combined_grape

# CLIP ViT-L/14
FAME_L_CHECKPOINT=/absolute/path/to/fashionclip_final.pth \
CUDA_VISIBLE_DEVICES=3 bash 命令.sh test_combined_grape_L
```
