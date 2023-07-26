# 🥳 Quictour

本快速浏览将向您展示 🤗 PEFT 的主要功能，并帮助您训练通常在消费设备上无法访问的大型预训练模型。您将了解如何使用[`bigscience/mt0-large`](https://huggingface.co/bigscience/mt0-large)LoRA 训练 1.2B 参数模型以生成分类标签并将其用于推理。

### PeftConfig

每个 🤗 PEFT 方法都由 PeftConfig 类定义，该类存储构建 PeftModel 的所有重要参数。

使用 LoRA，您需要加载并创建一个 LoraConfig 类。在 LoraConfig 中，指定以下参数：\


* the `task_type`, or sequence-to-sequence language modeling in this case 任务类型
* `inference_mode`, whether you’re using the model for inference or not 是否使用模型进行推理
* `r`, the dimension of the low-rank matrices 低秩矩阵的维度
* `lora_alpha`, the scaling factor for the low-rank matrice 低秩缩放因子
* `lora_dropout`, the dropout probability of the LoRA layers lora 层的丢失概率

```python
from peft import LoraConfig, TaskType
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
```

更多查看：[LoraConfig配置详情](https://huggingface.co/docs/peft/main/en/package\_reference/tuners#peft.LoraConfig)

PeftModel 由`get_peft_model()` 函数创建， [PeftConfig](https://huggingface.co/docs/peft/main/en/package\_reference/config#peft.PeftConfig) 中包含了如何为特定的PEFT方法配置模型的instruction.

**加载预训练模型**

<pre class="language-python"><code class="lang-python">from transformers import AutoModelForSeq2SeqLM
model_name_or_path = "bigscience/mt0-large" 
tokenizer_name_or_path = "bigscience/mt0-large" 
<strong>model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
</strong></code></pre>

