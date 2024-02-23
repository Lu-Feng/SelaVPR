# SelaVPR
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-seamless-adaptation-of-pre-trained/visual-place-recognition-on-pittsburgh-30k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-30k?p=towards-seamless-adaptation-of-pre-trained)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-seamless-adaptation-of-pre-trained/visual-place-recognition-on-pittsburgh-250k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-250k?p=towards-seamless-adaptation-of-pre-trained)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-seamless-adaptation-of-pre-trained/visual-place-recognition-on-tokyo247)](https://paperswithcode.com/sota/visual-place-recognition-on-tokyo247?p=towards-seamless-adaptation-of-pre-trained)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-seamless-adaptation-of-pre-trained/visual-place-recognition-on-mapillary-val)](https://paperswithcode.com/sota/visual-place-recognition-on-mapillary-val?p=towards-seamless-adaptation-of-pre-trained)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-seamless-adaptation-of-pre-trained/visual-place-recognition-on-nordland)](https://paperswithcode.com/sota/visual-place-recognition-on-nordland?p=towards-seamless-adaptation-of-pre-trained)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-seamless-adaptation-of-pre-trained/visual-place-recognition-on-st-lucia)](https://paperswithcode.com/sota/visual-place-recognition-on-st-lucia?p=towards-seamless-adaptation-of-pre-trained)

This is the official repository for the ICLR 2024 paper "[Towards Seamless Adaptation of Pre-trained Models for Visual Place Recognition](https://arxiv.org/pdf/2402.14505.pdf)".

This repo follows the [Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark). You can refer to it ([VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader)) to prepare datasets.

Download the pre-trained foundation model [DINOv2(ViT-L/14)](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth)

## Train

## Test

## Trained Models
The [model](https://drive.google.com/file/d/1FVb633PgbXsn-lASTANSGKQ46-vh3wla/view?usp=sharing) finetuned on MSLS (for diverse scenes).

The [model](https://drive.google.com/file/d/1kqOJaxHZbh6bSY4J1Hrymjg9S9ZxsrWv/view?usp=sharing) further finetuned on Pitts30k (only for urban scenes, e.g. Pitts30k, Pitss250k, and Tokyo24/7).

## Acknowledgements

## Citation
If you find this repo useful for your research, please consider citing the paper
```
@inproceedings{selavpr,
  title={Towards Seamless Adaptation of Pre-trained Models for Visual Place Recognition},
  author={Lu, Feng and Zhang, Lijun and Lan, Xiangyuan and Dong, Shuting and Wang, Yaowei and Yuan, Chun},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
