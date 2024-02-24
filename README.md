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
The model finetuned on MSLS (for diverse scenes).
<table>
<thead>
  <tr>
    <th rowspan="2">DOWNLOAD<br></th>
    <th colspan="3">MSLS-val</th>
    <th colspan="3">Nordland-test</th>
    <th colspan="3">St. Lucia</th>
  </tr>
  <tr>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th rowspan="3"><a href="https://drive.google.com/file/d/1FVb633PgbXsn-lASTANSGKQ46-vh3wla/view?usp=sharing">LINK</a></td>
    <td>90.8</td>
    <td>96.4</td>
    <td>97.2</td>
    <td>85.2</td>
    <td>95.5</td>
    <td>98.5</td>
    <td>99.8</td>
    <td>100.0</td>
    <td>100.0</td>
  </tr>
</tbody>
</table>

The model further finetuned on Pitts30k (only for urban scenes).
<table>
<thead>
  <tr>
    <th rowspan="2">DOWNLOAD<br></th>
    <th colspan="3">Tokyo24/7</th>
    <th colspan="3">Pitts30k</th>
    <th colspan="3">Pitts250k</th>
  </tr>
  <tr>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th rowspan="3"><a href="https://drive.google.com/file/d/1kqOJaxHZbh6bSY4J1Hrymjg9S9ZxsrWv/view?usp=sharing">LINK</a></td>
    <td>94.0</td>
    <td>96.8</td>
    <td>97.5</td>
    <td>92.8</td>
    <td>96.8</td>
    <td>97.7</td>
    <td>95.7</td>
    <td>98.8</td>
    <td>99.2</td>
  </tr>
</tbody>
</table>


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
