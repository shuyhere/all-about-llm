---
description: https://github.com/hiyouga/LLaMA-Efficient-Tuning/tree/main
---

# 🦈 LLaMA-Efficient-Tuning\&text-generation-webui

**仓库地址**：[https://github.com/hiyouga/LLaMA-Efficient-Tuning/tree/main](https://github.com/hiyouga/LLaMA-Efficient-Tuning/tree/main)

**可视化：**[https://github.com/oobabooga/text-generation-webui/tree/main](https://github.com/oobabooga/text-generation-webui/tree/main)



### 模型

* [LLaMA](https://github.com/facebookresearch/llama) (7B/13B/33B/65B)
* [LLaMA-2](https://huggingface.co/meta-llama) (7B/13B/70B)
* [BLOOM](https://huggingface.co/bigscience/bloom) & [BLOOMZ](https://huggingface.co/bigscience/bloomz) (560M/1.1B/1.7B/3B/7.1B/176B)
* [Falcon](https://huggingface.co/tiiuae/falcon-7b) (7B/40B)
* [Baichuan](https://huggingface.co/baichuan-inc/baichuan-7B) (7B/13B)
* [InternLM](https://github.com/InternLM/InternLM) (7B)

### 微调方法

* [二次预训练](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language\_understanding\_paper.pdf)
  * 全参数微调
  * 部分参数微调
  * [LoRA](https://arxiv.org/abs/2106.09685)
  * [QLoRA](https://arxiv.org/abs/2305.14314)
* [指令监督微调](https://arxiv.org/abs/2109.01652)
  * 全参数微调
  * 部分参数微调
  * [LoRA](https://arxiv.org/abs/2106.09685)
  * [QLoRA](https://arxiv.org/abs/2305.14314)
* [人类反馈的强化学习（RLHF）](https://arxiv.org/abs/2203.02155)
  * [LoRA](https://arxiv.org/abs/2106.09685)
  * [QLoRA](https://arxiv.org/abs/2305.14314)

### 数据集

* 用于二次预训练:
  * [Wiki Demo (en)](https://github.com/hiyouga/LLaMA-Efficient-Tuning/blob/main/data/wiki\_demo.txt)
  * [RefinedWeb (en)](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)
  * [StarCoder (en)](https://huggingface.co/datasets/bigcode/starcoderdata)
  * [Wikipedia (en)](https://huggingface.co/datasets/olm/olm-wikipedia-20221220)
  * [Wikipedia (zh)](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
* 用于指令监督微调:
  * [Stanford Alpaca (en)](https://github.com/tatsu-lab/stanford\_alpaca)
  * [Stanford Alpaca (zh)](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
  * [GPT-4 Generated Data (en\&zh)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
  * [Open Assistant (multilingual)](https://huggingface.co/datasets/OpenAssistant/oasst1)
  * [Self-cognition (zh)](https://github.com/hiyouga/LLaMA-Efficient-Tuning/blob/main/data/self\_cognition.json)
  * [ShareGPT (zh)](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Chinese-instruction-collection)
  * [RefGPT (zh)](https://github.com/sufengniu/RefGPT)
  * [Guanaco Dataset (multilingual)](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)
  * [BELLE 2M (zh)](https://huggingface.co/datasets/BelleGroup/train\_2M\_CN)
  * [BELLE 1M (zh)](https://huggingface.co/datasets/BelleGroup/train\_1M\_CN)
  * [BELLE 0.5M (zh)](https://huggingface.co/datasets/BelleGroup/train\_0.5M\_CN)
  * [BELLE Dialogue 0.4M (zh)](https://huggingface.co/datasets/BelleGroup/generated\_chat\_0.4M)
  * [BELLE School Math 0.25M (zh)](https://huggingface.co/datasets/BelleGroup/school\_math\_0.25M)
  * [BELLE Multiturn Chat 0.8M (zh)](https://huggingface.co/datasets/BelleGroup/multiturn\_chat\_0.8M)
  * [Firefly 1.1M (zh)](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
  * [CodeAlpaca 20k (en)](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)
  * [Alpaca CoT (multilingual)](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)
  * [Web QA (zh)](https://huggingface.co/datasets/suolyer/webqa)
  * [UltraChat (en)](https://github.com/thunlp/UltraChat)
  * [WebNovel (zh)](https://huggingface.co/datasets/zxbsmk/webnovel\_cn)
* 用于奖励模型训练:
  * [HH-RLHF (en)](https://huggingface.co/datasets/Anthropic/hh-rlhf)
  * [Open Assistant (multilingual)](https://huggingface.co/datasets/OpenAssistant/oasst1)
  * [GPT-4 Generated Data (en\&zh)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)

使用方法请参考 [data/README.md](https://github.com/hiyouga/LLaMA-Efficient-Tuning/blob/main/data/README\_zh.md) 文件。

部分数据集的使用需要确认，我们推荐使用下述命令登录您的 Hugging Face 账户。

```
pip install --upgrade huggingface_hub
huggingface-cli login
```

### 软件依赖

* Python 3.8+ 和 PyTorch 1.13.1+
* 🤗Transformers, Datasets, Accelerate, PEFT 和 TRL
* jieba, rouge-chinese 和 nltk (用于评估)
* gradio 和 matplotlib (用于网页端交互)
* uvicorn, fastapi 和 sse-starlette (用于 API)

以及 **强而有力的 GPU**！

### 如何使用

#### 数据准备（可跳过）

关于数据集文件的格式，请参考 `data/example_dataset` 文件夹的内容。构建自定义数据集时，既可以使用单个 `.json` 文件，也可以使用一个[数据加载脚本](https://huggingface.co/docs/datasets/dataset\_script)和多个文件。

注意：使用自定义数据集时，请更新 `data/dataset_info.json` 文件，该文件的格式请参考 `data/README.md`。

#### 环境搭建（可跳过）

```
git clone https://github.com/hiyouga/LLaMA-Efficient-Tuning.git
conda create -n llama_etuning python=3.10
conda activate llama_etuning
cd LLaMA-Efficient-Tuning
pip install -r requirements.txt
```

如果要在 Windows 平台上开启量化 LoRA（QLoRA），需要安装预编译的 `bitsandbytes` 库, 支持 CUDA 11.1 到 12.1.

```
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.39.1-py3-none-win_amd64.whl
```

#### 浏览器一键微调/测试

```
python src/train_web.py
```

目前网页 UI 仅支持单卡训练。

#### 二次预训练

```
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage pt \
    --model_name_or_path path_to_your_model \
    --do_train \
    --dataset wiki_demo \
    --finetuning_type lora \
    --output_dir path_to_pt_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
```

#### 指令监督微调

```
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_your_model \
    --do_train \
    --dataset alpaca_gpt4_en \
    --finetuning_type lora \
    --output_dir path_to_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
```

#### 奖励模型训练

```
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage rm \
    --model_name_or_path path_to_your_model \
    --do_train \
    --dataset comparison_gpt4_en \
    --finetuning_type lora \
    --output_dir path_to_rm_checkpoint \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16
```

#### RLHF 训练

```
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage ppo \
    --model_name_or_path path_to_your_model \
    --do_train \
    --dataset alpaca_gpt4_en \
    --finetuning_type lora \
    --checkpoint_dir path_to_sft_checkpoint \
    --reward_model path_to_rm_checkpoint \
    --output_dir path_to_ppo_checkpoint \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --resume_lora_training False \
    --plot_loss
```

#### 多 GPU 分布式训练

```
accelerate config # 首先配置分布式环境
accelerate launch src/train_bash.py # 参数同上
```

<details>

<summary>使用 DeepSpeed ZeRO-2 进行全参数微调的 Accelerate 配置示例</summary>

```
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 4
  gradient_clipping: 0.5
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</details>

#### 指标评估（BLEU分数和汉语ROUGE分数）

```
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_your_model \
    --do_eval \
    --dataset alpaca_gpt4_en \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_eval_result \
    --per_device_eval_batch_size 8 \
    --max_samples 100 \
    --predict_with_generate
```

我们建议在量化模型的评估中使用 `--per_device_eval_batch_size=1` 和 `--max_target_length 128` 参数。

#### 模型预测

```
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_your_model \
    --do_predict \
    --dataset alpaca_gpt4_en \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_predict_result \
    --per_device_eval_batch_size 8 \
    --max_samples 100 \
    --predict_with_generate
```

如果需要预测的样本没有标签，请首先在 `response` 列中填入一些占位符，以免样本在预处理阶段被丢弃。

#### API 服务

```
python src/api_demo.py \
    --model_name_or_path path_to_your_model \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint
```

关于 API 文档请见 `http://localhost:8000/docs`。

#### 命令行测试

```
python src/cli_demo.py \
    --model_name_or_path path_to_your_model \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint
```

#### 浏览器测试

```
python src/web_demo.py \
    --model_name_or_path path_to_your_model \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint
```

#### 导出微调模型

```
python src/export_model.py \
    --model_name_or_path path_to_your_model \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_export
```
