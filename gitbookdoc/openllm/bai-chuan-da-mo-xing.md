---
description: https://github.com/baichuan-inc
---

# 🌊 百川大模型

{% embed url="https://github.com/baichuan-inc/Baichuan-7B" %}

{% embed url="https://github.com/baichuan-inc/Baichuan-13B" %}

{% embed url="https://huggingface.co/baichuan-inc/Baichuan-13B-Chat" %}

## 概述

发布团队：百川智能

家族：llama

能否商用：能

成员：baichuan-7B(官方目前没有发布chat版本) baichuan-13B baichuan13B-chat











## 公开benchmark榜单

### 中文评测

1. [C-Eval](https://cevalbenchmark.com/index.html#home) 是一个全面的中文基础模型评测数据集，涵盖了 52 个学科和四个难度的级别。baichuan用该数据集的 dev 集作为 few-shot 的来源，在 test 集上进行了 `5-shot` 测试。

脚本： [https://github.com/baichuan-inc/Baichuan-7B/blob/main/evaluation/evaluate\_zh.py](https://github.com/baichuan-inc/Baichuan-7B/blob/main/evaluation/evaluate\_zh.py)

2. [Gaokao](https://github.com/OpenLMLab/GAOKAO-Bench) 是一个以中国高考题作为评测大语言模型能力的数据集，用以评估模型的语言能力和逻辑推理能力。 baichuan只保留了其中的单项选择题，随机划分后对所有模型进行统一 `5-shot` 测试。
3.  [AGIEval](https://github.com/microsoft/AGIEval) 旨在评估模型的认知和解决问题相关的任务中的一般能力。 我们只保留了其中的四选一单项选择题，随机划分后对所有模型进行了统一 `5-shot` 测试。



### 英文评测

[MMLU](https://arxiv.org/abs/2009.03300) 是包含 57 个多选任务的英文评测数据集，涵盖了初等数学、美国历史、计算机科学、法律等，难度覆盖高中水平到专家水平，是目前主流的LLM评测数据集。 baichuan使用`5-shot`评估

评测方案：[https://github.com/hendrycks/test](https://github.com/hendrycks/test)\


## Third-Party Resources

1. [LLaMA Efficient Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) 支持Baichuan-7B使用Qlora进行Finetune，支持RLHF，支持WebDemo。使用经过sft的模型见 [hiyouga/baichuan-7b-sft](https://huggingface.co/hiyouga/baichuan-7b-sft)。
2. [fireballoon/baichuan-vicuna-chinese-7b](https://huggingface.co/fireballoon/baichuan-vicuna-chinese-7b) 使用 ShareGPT, ShareGPT-ZH, COT & COT-ZH, Leetcode, dummy等包含中英文的数据Finetune后的模型，训练代码参考FastChat。
3. [fireballoon/baichuan-vicuna-7b](https://huggingface.co/fireballoon/baichuan-vicuna-7b) 使用ShareGPT, COT 和 Leetcode等数据混合Finetune后的模型，训练代码参考FastChat。
4. [Efficient-Tuning-LLMs](https://github.com/jianzhnie/Efficient-Tuning-LLMs) 支持Baichuan-7B使用Qlora进行Finetune和4bit inference。
5. [fastllm](https://github.com/ztxz16/fastllm) fastllm是纯c++实现，无第三方依赖的大模型库，支持Baichuan-7B在手机端运行。
6. [TheBloke/baichuan-7B-GPTQ](https://huggingface.co/TheBloke/baichuan-7B-GPTQ) 对Baichuan-7B的GPTQ 4bit量化。

## 自己试跑遇到的问题

7B预训练模型续写的句子:

{'text': '今天天气是真的有点热,我走在街上的时候,发现了很多人的脸上泛了红色的......\n我走在街上的时候,发现了很多人的脸上泛了红色的,我问老爸:为什么现在有这么多人的脸上都长了斑呢?老爸说:因为天气太热了,脸上没有汗水的滋润,皮肤没有光泽了。<mark style="background-color:red;">你别问了,我现在正在去开会的地方的路上。</mark>'}

7b预计在A100占用显存 ：27689MB ； 4090占用显存23.6G  推理速度很慢
