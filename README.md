# RAVAR (Referring Atomic Video Action Recognition ECCV 2024)--Releasement in Progress

![](https://img.shields.io/badge/version-1.0.1-blue)
[![arxiv badge](https://img.shields.io/badge/arxiv-red)](TODO)
[![ECCV](https://img.shields.io/badge/ECCV-2024-%23f1592a?labelColor=%23003973&color=%23be1c1a)](TODO)

### Kunyu Peng*, Jia Fu*, Kailun Yang°, Di Wen, Yufan Chen, Ruiping Liu, Junwei Zheng, Jiaming Zhang, M. Saquib Sarfraz, Rainer Stiefelhagen, and Alina Roitberg

>We introduce a new task called Referring Atomic Video Action Recognition (RAVAR), aimed at identifying atomic actions of a particular person based on a textual description and the video data of this person. This task differs from traditional action recognition and localization, where predictions are delivered for all present individuals. In contrast, we focus on recognizing the correct atomic action of a specific individual, guided by text. To explore this task, we present the RefAVA dataset, containing 36,630 instances with manually annotated textual descriptions of the individuals. To establish a strong initial benchmark, we implement and validate baselines from various domains, e.g., atomic action localization, video question answering, and text-video retrieval. Since these existing methods underperform on RAVAR, we introduce RefAtomNet -- a novel cross-stream attention-driven method specialized for the unique challenges of RAVAR: the need to interpret a textual referring expression for the targeted individual, utilize this reference to guide the spatial localization and harvest the prediction of the atomic actions for the referring person. The key ingredients are: (1) a multi-stream architecture that connects video, text, and a new location-semantic stream, and (2) cross-stream agent attention fusion and agent token fusion which amplify the most relevant information across these streams and consistently surpasses standard attention-based fusion on RAVAR. Extensive experiments demonstrate the effectiveness of RefAtomNet and its building blocks for recognizing the action of the described individual. The dataset and code will be made publicly available. (* indicates shared first author, ° indicates corresponding author)

- Due to the **```page and format restrictions```** set by ECCV publications, we have omitted some details and appendix content. For the complete version of the paper, including the **```selection of prompts```** and **```experiment details```**, please refer to our [arXiv version]([TODO](TODO)).

## 🤖 Model Architecture
![Model_architecture](https://github.com/KPeng9510/RAVAR/blob/main/main.png)

## 📈 Results
<div align="center">
<img src="https://github.com/KPeng9510/RAVAR/blob/main/results.png" width="70%" />
</div>

## 📚 Dataset Download
- Link (TODO)

## 🎨 Training & Testing

### Training

TODO
## 📕 Installation

- Python >= 3.8
- PyTorch >= 1.9.0
- PyYAML, tqdm, tensorboardX


## 🤝 Cite:
Please consider citing this paper if you use the ```code``` or ```data``` from our work.
Thanks a lot :)

```
@inproceedings{peng2024ravar,
  title={Referring Atomic Video Action Recognition},
  author={Kunyu Peng and Jia Fu and Kailun Yang and Di Wen and Yufan Chen and Ruiping Liu and Junwei Zheng and Jiaming Zhang and M. Saquib Sarfraz and Rainer Stiefelhagen and Alina Roitberg},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
