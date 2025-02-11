export CUDA_VISIBLE_DEVICES="5"
export MODLE="stabilityai/stable-diffusion-2-1-base"
export SOURCE_IMAGE="./example/example_image.jpg"
export OUTPUT_DIR="./example"
python get_bbox.py \
    --model_id $MODLE \
    --source_image $SOURCE_IMAGE \
    --source_prompt "a person holding a shell in front of the ocean" \
    --guidance_scale 3 \
    --word_idx 5 \
    --output $OUTPUT_DIR \
    --iters 3
