#!/bin/bash

# Check if root directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <root_directory>"
    exit 1
fi

root=$1

# Remove trailing slash if present in root
root=${root%/}

echo "Processing video..."
python scripts/process_in_the_wild_video.py --input_dir $root/raw/ --pad_instead_of_crop

cd ../MODNet
# Process each folder in the frames directory
echo "Processing masks..."
for folder in $root/frames/*/; do
    folder_name=$(basename "$folder")
    output_dir="$root/mask/$folder_name"
    mkdir -p "$output_dir"
    
    python inference_MODNet.py \
        --input-path "$folder" \
        --output-path "$output_dir" \
        --ckpt-path ./pretrained/modnet_photographic_portrait_matting.ckpt
done
cd ../GVFDiffusion

echo "Encoding frames..."
python scripts/encode_in_the_wild_img_cond_dinov2_feature.py --input_dir $root/frames/ --start_frame_file $root/video_names.txt

# Count the number of videos (directories) in frames folder
num_videos=$(($(find "$root/frames/resized_images" -maxdepth 1 -type d | grep -v "^$root/frames/resized_images$" | wc -l) - 0))

echo "Running accelerate launch with $num_videos samples..."
accelerate launch --num_processes 1 inference_dpm_latent.py --batch_size 1 --exp_name ${root}_infer_ema1300k_video  --config configs/diffusion.yml --start_idx 0 --end_idx $num_videos --txt_file $root/video_names.txt --use_fp16  --num_samples $num_videos --adaptive --data_dir $root  --num_timesteps 32  --in_the_wild
fi
