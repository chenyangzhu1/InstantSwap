export CUDA_VISIBLE_DEVICES="0"
export MODLE="path/to/flower_checkpoint"
export SOURCE_MASK="./example/bbox.jpg"
export SOURCE_IMAGE="./example/example_image.jpg"
export OUTPUT_DIR="./example/flower"
python InstantSwap.py \
    --model_id $MODLE \
    --source_mask $SOURCE_MASK \
    --source_image $SOURCE_IMAGE \
    --source_prompt "a person holding a shell in front of the ocean" \
    --target_prompt "a person holding a sks flower in front of the ocean" \
    --diff_prompt "sks flower" \
    --diff_prompt_source "shell" \
    --guidance_scale 7.5 \
    --output $OUTPUT_DIR \
    --interval 5 \
    --iters 550 
