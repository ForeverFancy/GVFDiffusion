# Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis [ICCV 2025]

By [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen), [Sicheng Xu](https://github.com/sicxu), [Chuxin Wang](https://chuxwa.github.io/), [Jiaolong Yang](https://jlyang.org/), [Feng Zhao](https://en.auto.ustc.edu.cn/2021/0616/c26828a513169/page.htm), [Dong Chen](http://www.dongchen.pro/), [Baining Guo](https://www.microsoft.com/en-us/research/people/bainguo/).

[Paper](https://arxiv.org/abs/2507.23785) | [Project Page](https://gvfdiffusion.github.io/) | [Code](https://github.com/ForeverFancy/GVFDiffusion)

## Abstract

We present a novel framework for video-to-4D generation that creates high-quality dynamic 3D content from single video inputs. Direct 4D diffusion modeling is extremely challenging due to costly data construction and the high-dimensional nature of jointly representing 3D shape, appearance, and motion. We address these challenges by introducing a *Direct 4DMesh-to-GS Variation Field VAE* that directly encodes canonical Gaussian Splats (GS) and their temporal variations from 3D animation data without per-instance fitting, and compresses high-dimensional animations into a compact latent space. Building upon this efficient representation, we train a *Gaussian Variation Field diffusion model* with temporal-aware Diffusion Transformer conditioned on input videos and canonical GS. Trained on carefully-curated animatable 3D objects from the Objaverse dataset, our model demonstrates superior generation quality compared to existing methods. It also exhibits remarkable generalization to in-the-wild video inputs despite being trained exclusively on synthetic data, paving the way for generating high-quality animated 3D content.

## Installation

```bash
git clone https://github.com/BwZhang/GVFDiffusion.git
cd GVFDiffusion
. ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```

## Quick Start with Minimal Example

```bash
accelerate launch --num_processes 1 inference_dpm_latent.py --batch_size 1 --exp_name /path/to/your/output  --config configs/diffusion.yml --start_idx 0 --end_idx 2 --txt_file ./assets/in_the_wild.txt --use_fp16 --num_samples 2 --adaptive --data_dir ./assets/  --num_timesteps 32 --download_assets --in_the_wild
```

## Inference In-the-Wild Data

Clone the MODNet repository, link the inference script and download the pre-trained model to `./pretrained/modnet_photographic_portrait_matting.ckpt`.

```bash
cd ..
git clone https://github.com/ZHKKKe/MODNet.git
ln -s MODNet/scripts/inference_MODNet.py GVFDiffusion/scripts/inference_MODNet.py
```

```bash
cd GVFDiffusion
. ./scripts/run_in_the_wild.sh /path/to/your/in_the_wild_video_folder
```

## Data Preparation

### 4D Data Rendering

We mainly follow the data rendering scripts in [Diffusion4D](https://github.com/VITA-Group/Diffusion4D) with some modification.

For Objaverse-1.0 data rendering, run the following commands:
```bash
cd data_rendering/rendering
python download.py --start_idx 0 --end_idx 9000
python render.py --obj_path "./obj_v1/glbs" --save_dir /mnt/blob/data/objaverse_4d_rendering_no_light --gpu_num 4
```

For Objaverse-XL data rendering, first download the [metadata](https://huggingface.co/BwZhang/GaussianVariationFieldDiffusion/blob/main/data/metadata.csv) to `./data/objaverse_4d_rendering/` and then run the following commands:
```bash
cd data_rendering/rendering
python download_xl.py ObjaverseXL --output_dir ./data/objaverse_4d_rendering/ --start_idx 0 --end_idx 25000
python render_xl.py ObjaverseXL --output_dir ./data/objaverse_4d_rendering/ --start_idx 0 --end_idx 25000 --num_gpus 8 --max_workers 16
```

### VAE Training Data Preparation

We mainly follow the data preparation process of [TRELLIS](https://github.com/microsoft/TRELLIS/blob/main/DATASET.md) for data rendering, mesh voxelization, DINOv2 feature reprojection and encoding.

### Conditional Features for Diffusion Training

```bash
python scripts/encode_img_cond_dinov2_feature.py --input_dir ./data/objaverse_4d_rendering/ --output_dir ./data/objaverse_4d_data/dinov2_features --txt_file ./assets/4d_objs.txt --end_idx 34000 --gpus 8
```

## Training

### VAE Training

Download initial TRELLIS VAE checkpoint from [here]() and put it in the `./checkpoints/` folder. Download the data list from [here](https://huggingface.co/BwZhang/GaussianVariationFieldDiffusion/blob/main/assets/4d_objs.txt) and put it in the `./assets/` folder.

```bash
accelerate launch --num_processes 8  main_vae.py --log_interval 100 --batch_size 2 --lr 5e-5 --weight_decay 0 --exp_name /path/to/your/output --save_interval 5000 --config configs/vae.yml --use_tensorboard --use_vgg --load_camera 1 --start_idx 0 --end_idx 9000 --txt_file ./assets/4d_objs.txt  --use_fp16 --data_dir ./data/objaverse_4d_rendering/ --kl_weight 1e-6 --render_l1_weight 1.0 --render_lpips_weight 0.2 --render_ssim_weight 0.2 --xyz_loss_weight 1.0 --gradient_accumulation_steps 2 --static_vae_steps 150000  --static_vae_ckpt ./checkpoints/trellis_init_vae.ckpt
```

### Diffusion Training

First, encode the latent codes using VAE:

```bash
accelerate launch --num_processes 8 encode_latent.py  --batch_size 1 --exp_name ./data/objaverse_4d_data/latents --config configs/vae.yml --start_idx 0 --end_idx 34000 --txt_file ./assets/4d_objs.txt --use_fp16 --data_dir ./data/objaverse_4d_rendering/ --ckpt /path/to/deformation_checkpoint.pt --static_vae_ckpt /path/to/static_checkpoint.pt --num_samples 34000
```

Download the mean and std files from [here](https://huggingface.co/BwZhang/GaussianVariationFieldDiffusion/tree/main) and put them in the `./checkpoints/` folder.

```bash
accelerate launch --num_processes 8 main_latent.py --log_interval 100 --batch_size 2 --lr 5e-5 --weight_decay 0 --exp_name /path/to/your/output --save_interval 5000 --config configs/diffusion.yml --use_tensorboard --start_idx 0 --end_idx 34000 --txt_file ./assets/4d_objs.txt  --use_fp16 --data_dir ./data/objaverse_4d_data/ --uncond_p 0.1  --sample_timesteps 24 --gradient_accumulation_steps 2 --deformation_mean_file ./checkpoints/deformation_mean.pt --deformation_std_file ./checkpoints/deformation_std.pt --static_mean_file ./checkpoints/static_mean.pt --static_std_file ./checkpoints/static_std.pt --ckpt /path/to/checkpoint.pt
```

## Acknowledgement

This codebase is built upon [TRELLIS](), and [3dshape2vecset](). Thanks authors for their great work. Also thank the authors of [Diffusion4D]() for the data rendering scripts.

## Citation

If you find the work useful, please consider citing:
```
@misc{zhang2025gaussianvariationfielddiffusion,
        title={Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis}, 
        author={Bowen Zhang and Sicheng Xu and Chuxin Wang and Jiaolong Yang and Feng Zhao and Dong Chen and Baining Guo},
        year={2025},
        eprint={2507.23785},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2507.23785}, 
      }
```
