---
description: https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila
---

# 🦅 Aquila 悟道天鹰系列

发布团队：智源

能否商用：能

成员：

<table><thead><tr><th width="166">模型</th><th width="129">模型类型</th><th width="318">简介</th><th>文件路径</th><th>单独下载模型权重</th><th>状态</th><th>训练所用显卡</th></tr></thead><tbody><tr><td>Aquila-7B</td><td>基础模型，70亿参数</td><td><strong>Aquila 基础模型</strong>在技术上继承了 GPT-3、LLaMA 等的架构设计优点，替换了一批更高效的底层算子实现、重新设计实现了中英双语的 tokenizer，升级了 BMTrain 并行训练方法，实现了比 Magtron+DeepSpeed ZeRO-2 将近８倍的训练效率。</td><td><a href="https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila/Aquila-pretrain">./examples/Aquila/Aquila-pretrain</a></td><td><a href="http://model.baai.ac.cn/model-detail/100098">下载Aquila-7B</a></td><td>已发布</td><td>Nvidia-A100</td></tr><tr><td>Aquila-33B</td><td>基础模型，330亿参数</td><td>同上</td><td>——</td><td>——</td><td><strong>敬请期待</strong></td><td>Nvidia-A100</td></tr><tr><td>AquilaChat-7B</td><td>SFT model，基于 Aquila-7B 进行微调和强化学习</td><td><strong>AquilaChat 对话模型</strong>支持流畅的文本对话及多种语言类生成任务，通过定义可扩展的特殊指令规范，实现 AquilaChat对其它模型和工具的调用，且易于扩展。<br><br>例如，调用智源开源的 <a href="https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltDiffusion-m18"><strong>AltDiffusion</strong></a> <strong>多语言文图生成模型</strong>，实现了流畅的文图生成能力。配合智源 <strong>InstructFace 多步可控文生图模型</strong>，轻松实现对人脸图像的多步可控编辑。</td><td><a href="https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila/Aquila-chat">./examples/Aquila/Aquila-chat</a></td><td><a href="https://model.baai.ac.cn/model-detail/100101">下载AquilaChat-7B</a></td><td>已发布</td><td>Nvidia-A100</td></tr><tr><td>AquilaChat-33B</td><td>SFT model，基于 Aquila-33B 进行微调和强化学习</td><td>同上</td><td>——</td><td>——</td><td><strong>敬请期待</strong></td><td>Nvidia-A100</td></tr><tr><td>AquilaCode-7B-NV</td><td>基础模型，“文本-代码”生成模型，基于 Aquila-7B继续预训练，在英伟达芯片完成训练</td><td>AquilaCode-7B 以小数据集、小参数量，实现高性能，是目前支持中英双语的、性能最好的开源代码模型，经过了高质量过滤、使用有合规开源许可的训练代码数据进行训练。<br><br>AquilaCode-7B 分别在英伟达和国产芯片上完成了代码模型的训练。</td><td><a href="https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila/Aquila-code">./examples/Aquila/Aquila-code</a></td><td><a href="https://model.baai.ac.cn/model-detail/100102">下载AquilaCode-7B-NV</a></td><td>已发布</td><td>Nvidia-A100</td></tr><tr><td>AquilaCode-7B-TS</td><td>基础模型，“文本-代码”生成模型，基于 Aquila-7B继续预训练，在天数智芯芯片上完成训练</td><td>同上</td><td><a href="https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila/Aquila-code">./examples/Aquila/Aquila-code</a></td><td><a href="https://model.baai.ac.cn/model-detail/100099">下载AquilaCode-7B-TS</a></td><td>已发布</td><td>Tianshu-BI-V100</td></tr></tbody></table>

ps：悟道·天鹰Aquila系列模型将持续开源更优版本，先删除原来目录下的`checkpoints_in/aquila-7b`，再下载新权重，其他使用方式不变。

* 2023/07/13 ：发布权重文件 v0.8，开源了 Aquila-7B、AquilaChat-7B 最新权重，AquilaCode 权重无更新。
* [变更日志](https://github.com/FlagAI-Open/FlagAI/blob/master/examples/Aquila/changelog.md)

## 支持BMInf高效推理

[https://github.com/OpenBMB/BMInf/blob/master/README-ZH.md](https://github.com/OpenBMB/BMInf/blob/master/README-ZH.md)

##

## 评测

[FlagEval （天秤）大模型评测体系及开放平台](https://flageval.baai.ac.cn/#/home)



##

##

##

## Issues

* [OOM问题](https://github.com/FlagAI-Open/FlagAI/issues/334)

