<div align="center">
<h2><font> InstantSwap: </font></center> <br> <center>Fast Customized Concept Swapping across Sharp Shape Differences</h2>

[Chenyang Zhu](https://chenyangzhu1.github.io/), [Kai Li](https://kailigo.github.io/), [Yue Ma](https://mayuelala.github.io/), [Longxiang Tang](https://github.com/chenyangzhu1/InstantSwap), [Chengyu Fang](https://chunminghe.github.io/), [Chubin Chen](https://github.com/chenyangzhu1/InstantSwap), [Qifeng Chen](https://cqf.io/) and [Xiu Li](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ&hl=zh-CN&oi=sra)

<strong>ICLR 2025</strong>

<a href='https://arxiv.org/abs/2412.01197'><img src='https://img.shields.io/badge/ArXiv-2412.01197-red'></a>
<a href='https://instantswap.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
![visitors](https://visitor-badge.laobi.icu/badge?page_id=chenyangzhu1.InstantSwap)

</div>

![results](figs/title_case-small.jpg "results")

## ⚡️ Abstract

Recent advances in Customized Concept Swapping (CCS) enable a text-to-image model to swap a concept in the source image with a customized target concept.
However, the existing methods still face the challenges of **_inconsistency_** and **_inefficiency_**. They struggle to maintain consistency in both the foreground and background during concept swapping, especially when the shape difference is large between objects.
Additionally, they either require time-consuming training processes or involve redundant calculations during inference.
To tackle these issues, we introduce InstantSwap, a new CCS method that aims to handle sharp shape disparity at speed.
Specifically, we first extract the bbox of the object in the source image _automatically_ based on attention map analysis and leverage the bbox to achieve both foreground and background consistency. For background consistency, we remove the gradient outside the bbox during the swapping process so that the background is free from being modified.
For foreground consistency, we employ a cross-attention mechanism to inject semantic information into both source and target concepts inside the box.
This helps learn semantic-enhanced representations that encourage the swapping process to focus on the foreground objects.
To improve swapping speed, we avoid computing gradients at each timestep but instead calculate them periodically to reduce the number of forward passes, which improves efficiency a lot with a little sacrifice on performance.
Finally, we establish a benchmark dataset to facilitate comprehensive evaluation. Extensive evaluations demonstrate the superiority and versatility of InstantSwap.

## 📣 Updates

- **[2025.2.11]** 🔥 Release the Inference Code and ConSwapBench.
- **[2025.1.23]** 🔥 InstantSwap is accepted by ICLR 2025!
- **[2024.12.2]** 🔥 Release Paper and Project page!

## 🚧 Todo
We are working hard to clean up the code and will open source it as soon as we get permission.
- [x] Release Inference Code
- [x] Release ConSwapBench
- [ ] Release Evaluation Code

## 🚩 Getting Started

Next, this repo will show you how to use InstantSwap to swap the shell 🐚 in your hand with a beautiful flower🌹.
The example image is in the `example` folder, which is downloaded from [Unsplash](https://unsplash.com/photos/a-hand-holding-a-rock-near-the-ocean-XT4iRGoIAwQ).

The following is a step-by-step guide:

### ⚙️ Environment Setup
You can run this [bash script][setup] or use the following command to obtain the bounding box of the image.
```shell
conda create -n ISwap python=3.9
conda activate ISwap
pip install -r requirements.txt
```

### 1. Customized Concept Learning

The new subject will be integrated as a novel token (e.g. `sks`) within the diffusion model. Users can either utilize the [DreamBooth scripts](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) to learn the new concept or directly download the [checkpoint](https://huggingface.co/datasets/zcaoyao/Flower_Concept) for the flower concept.

### 2. Obtain bbox (optional)


You can run this [bash script][get_bbox] or use the following command to obtain the bounding box of the image.

```shell
export MODLE="stabilityai/stable-diffusion-2-1-base"
export SOURCE_IMAGE="./example_image.jpg"
export OUTPUT_DIR="./example"
python get_bbox.py \
    --model_id $MODLE \
    --source_image $SOURCE_IMAGE \
    --source_prompt "a person holding a shell in front of the ocean" \
    --guidance_scale 3 \
    --word_idx 5 \
    --output $OUTPUT_DIR \
    --iters 3
```

### 3. Run InstantSwap

Finally, you can run this [bash script][InstantSwap] or use the following command to swap the shell in your hand for a flower.

```shell
export MODLE="path/to/the/Flower_Concept_Model"
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
```

## ConSwapBench 🧾
You can download and use the complete [ConSwapBench](https://huggingface.co/datasets/zcaoyao/ConSwapBench).

## Results 🎉

More results can be found in our [Project page](https://instantswap.github.io/).

![results](figs/quality_comparison-small.jpg "compare")

## Citation 📄
```bibtex
@article{zhu2024instantswap,
  title={Instantswap: Fast customized concept swapping across sharp shape differences},
  author={Zhu, Chenyang and Li, Kai and Ma, Yue and Tang, Longxiang and Fang, Chengyu and Chen, Chubin and Chen, Qifeng and Li, Xiu},
  journal={arXiv preprint arXiv:2412.01197},
  year={2024}
}
```

## Acknowledgement 🙏

This repository borrows heavily from [Prompt-to-prompt](https://github.com/google/prompt-to-prompt) and 🤗[Diffusers](https://huggingface.co/docs/diffusers/main/en/index). Thanks to the authors for sharing their code and models.

## Contact ✉️
This is the codebase for our research work. We are still working hard to update this repo, and more details are coming in days. If you have any questions or ideas to discuss, feel free to contact [Chenyang Zhu](chenyangzhu.cs@gmail.com).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=chenyangzhu1/InstantSwap&type=Date)](https://star-history.com/#chenyangzhu1/InstantSwap&Date)

[get_bbox]: get_bbox.sh
[InstantSwap]: InstantSwap.sh
[setup]: setup_env.sh
