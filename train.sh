export MODEL_DIR='/home/zs/disk/weights/AI-ModelScope/stable-diffusion-v1-5'
export OUTPUT_DIR="output"

##export PYTHONPATH=$(pwd):$PYTHONPATH
##--include localhost:0
#CUDA_VISIBLE_DEVICES=4,5,6
accelerate launch --mixed_precision="fp16" train.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir='/home/zs/disk/ca/mmd/temp/m' \
 --val_data_dir='/home/zs/disk/ca/mmd/temp/m' \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" \
 --validation_prompt ""\
 --train_batch_size=2 \
 --gradient_accumulation_steps=2 \
 --lora_rank 1 \
 --fusion_blocks "full" \
 --num_train_epochs 1000 \
# --use_8bit_adam

#--multi_gpu