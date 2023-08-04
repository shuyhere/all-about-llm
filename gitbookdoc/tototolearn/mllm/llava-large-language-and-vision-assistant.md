---
description: 针对具有 GPT-4 级别功能的大型语言和视觉模型进行视觉指令调整。
---

# 🐐 LLaVA: Large Language and Vision Assistant

[github](https://github.com/haotian-liu/LLaVA/)

[weights](https://huggingface.co/liuhaotian)

## 训练资源消耗

LLaVA 训练由两个阶段组成：（1）特征对齐阶段：使用大约 600K 过滤的 CC3M 将_冻结的预训练_视觉编码器连接到_冻结的 LLM_；(2)视觉指令调优阶段：使用150K GPT生成的多模态指令跟随来教导模型遵循多模态指令。

**LLaVA 在 8 个具有 80GB 内存的 A100 GPU 上进行训练**。要在更少的 GPU 上进行训练，您可以相应地减少`per_device_train_batch_size`和增加`gradient_accumulation_steps`。始终保持全局批量大小相同：`per_device_train_batch_size`x `gradient_accumulation_steps`。

