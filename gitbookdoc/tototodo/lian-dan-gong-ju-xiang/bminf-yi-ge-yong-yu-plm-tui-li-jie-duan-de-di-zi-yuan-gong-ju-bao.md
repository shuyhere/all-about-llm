---
description: https://github.com/OpenBMB/BMInf
---

# 🙆♂ BMInf -- 一个用于PLM推理阶段的低资源工具包

[BMInf(Big Model Inference)](https://github.com/OpenBMB/BMInf) 是一个用于大规模预训练语言模型（pretrained language models, PLM）推理阶段的低资源工具包。BMInf最低支持在NVIDIA GTX 1060单卡运行百亿大模型。在此基础上，使用更好的gpu运行会有更好的性能。在显存支持进行大模型推理的情况下（如V100或A100显卡），BMInf的实现较现有PyTorch版本仍有较大性能提升。

**安装：**

`pip install bminf`

**运行BMInf的最低配置与推荐配置：**

|       | 最低配置                        | 推荐配置                   |
| ----- | --------------------------- | ---------------------- |
| 内存    | 16GB                        | 24GB                   |
| GPU   | NVIDIA GeForce GTX 1060 6GB | NVIDIA Tesla V100 16GB |
| PCI-E | PCI-E 3.0 x16               | PCI-E 3.0 x16          |

### Reference

1. [CPM-2: Large-scale Cost-efficient Pre-trained Language Models.](https://arxiv.org/abs/2106.10715) Zhengyan Zhang, Yuxian Gu, Xu Han, Shengqi Chen, Chaojun Xiao, Zhenbo Sun, Yuan Yao, Fanchao Qi, Jian Guan, Pei Ke, Yanzheng Cai, Guoyang Zeng, Zhixing Tan, Zhiyuan Liu, Minlie Huang, Wentao Han, Yang Liu, Xiaoyan Zhu, Maosong Sun.
2. [CPM: A Large-scale Generative Chinese Pre-trained Language Model.](https://arxiv.org/abs/2012.00413) Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun.
3. [EVA: An Open-Domain Chinese Dialogue System with Large-Scale Generative Pre-Training.](https://arxiv.org/abs/2108.01547) Hao Zhou, Pei Ke, Zheng Zhang, Yuxian Gu, Yinhe Zheng, Chujie Zheng, Yida Wang, Chen Henry Wu, Hao Sun, Xiaocong Yang, Bosi Wen, Xiaoyan Zhu, Minlie Huang, Jie Tang.
4. [Language Models are Unsupervised Multitask Learners.](http://www.persagen.com/files/misc/radford2019language.pdf) Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever.

##

##
