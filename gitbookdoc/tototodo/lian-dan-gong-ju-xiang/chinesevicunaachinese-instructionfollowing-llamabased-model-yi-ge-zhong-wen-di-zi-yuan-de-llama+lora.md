---
description: llama+lora
---

# 🦙 Chinese-Vicuna: A Chinese Instruction-following LLaMA-based Model —— 一个中文低资源的llama+lora方案

[github](https://github.com/Facico/Chinese-Vicuna/tree/master)

<mark style="background-color:green;">git仓库提供了一个针对llama系列工程很好的QA文档：</mark>[<mark style="background-color:green;">https://github.com/Facico/Chinese-Vicuna/blob/master/docs/notes.md</mark>](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/notes.md)

### 概述

鉴于[llama](https://github.com/facebookresearch/llama),[alpaca](https://github.com/tatsu-lab/stanford\_alpaca),[guanaco](https://github.com/Guanaco-Model/Guanaco-Model.github.io)等羊驼模型的研发成功，本项目基于LLaMA+instruction数据构建一个中文的羊驼模型，并帮助大家能快速学会使用引入自己的数据，并训练出属于自己的小羊驼（Vicuna）

代码基于alpaca-lora开发，[https://github.com/tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)

参数高效，显卡友好，部署简易：

* 在一张2080Ti（11G）上可以对Llama-7B进行指令微调 ([7b-instruct](https://huggingface.co/Chinese-Vicuna/Chinese-Vicuna-lora-7b-belle-and-guanaco))
* 在一张3090（24G）上可以对Llama-13B进行指令微调 ([13b-instruct](https://huggingface.co/Chinese-Vicuna/Chinese-Vicuna-lora-13b-belle-and-guanaco))
* 即使是长度为2048的对话，在3090上也可以完成Llama-7B的微调；使用5万条数据即可有不错效果 ([chatv1](https://huggingface.co/Chinese-Vicuna/Chinese-Vicuna-lora-7b-chatv1))
* 领域微调的例子：医学问答 和 法律问答。([medical](https://huggingface.co/Chinese-Vicuna/Chinese-Vicuna-continue-finetune-7epoch-cMedQA2) and [legal](https://huggingface.co/Chinese-Vicuna/Chinese-Vicuna-7b-legal-lora))
* 支持`qlora-4bit`，使用4bit可以在2080Ti上完成13B的训练
* 可在2080Ti/3090上轻松部署，支持多卡同时推理，可进一步降低显存占用

项目包括

* finetune模型的代码
* 推理的代码
* 仅使用CPU推理的代码 (使用C++)
* 下载/转换/量化Facebook llama.ckpt的工具
* 其他应用

### 训练一个lora需要什么

#### 代码

* 此代码基于alpaca-lora开发，[https://github.com/tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)
* 这是一套比较简单的代码，基本思路就是用[PEFT](https://github.com/huggingface/peft)的lora接口+transformer的trainer+instruction的数据配置

#### 数据

这些数据很多都像alpaca那样，使用chatgpt的接口，生成高质量的instruction数据。

* [Belle](https://github.com/LianjiaTech/BELLE)
* [guanaco](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)

数据格式比较简单，基本如下：

* {% code fullWidth="true" %}
  ```
  {
  'instruction': 
  'input': 
  'output'
  }
  ```
  {% endcode %}

即需要一个指令，一个input，一个output。由于数据处理的时候是直接将instruction和input连接起来的，所以数据其实可以只需要instruction和output，如

* ```
  {
  'instruction': "用一句话描述地球为什么是独一无二的。\\n\n"
  'input': ""
  'output': "地球上有适宜生命存在的条件和多样化的生命形式。"
  }
  ```

<mark style="background-color:green;">数据下载：</mark>

* 链接: [https://pan.baidu.com/s/1WSxuhSAotl14ifaAiz5eKw?pwd=b4kb](https://pan.baidu.com/s/1WSxuhSAotl14ifaAiz5eKw?pwd=b4kb) 提取码: b4kb
* 链接: [https://drive.google.com/file/d/1tzXVhS74m-EtoFot7hEc005LDeZGPit\_/view?usp=sharing](https://drive.google.com/file/d/1tzXVhS74m-EtoFot7hEc005LDeZGPit\_/view?usp=sharing)
* 链接: [https://huggingface.co/datasets/Chinese-Vicuna/guanaco\_belle\_merge\_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco\_belle\_merge\_v1.0)

#### 上游模型

LLAMA-7B

#### lora模型

* 仓库提供了一个在上面混合数据上训练的lora模型
  * 你可以从huggingface上加载现成的模型或其他lora模型，加载方式参考[generate.py](https://github.com/Facico/Chinese-Vicuna/blob/master/generate.py)
    * `Chinese-Vicuna/Chinese-Vicuna-lora-7b-belle-and-guanaco`
    * `Chinese-Vicuna/Chinese-Vicuna-lora-13b-belle-and-guanaco`
  * 模型使用的是8bit+lora+256 tokens
  * 更多模型请查看：[https://huggingface.co/Chinese-Vicuna](https://huggingface.co/Chinese-Vicuna)

#### 设备

* 训练：<mark style="color:red;">一张2080Ti即可</mark>。由于数据长度都在256（代码设置为cutoff\_len，默认阶段长度）以内，大概占用9G显存。
  * <mark style="color:red;">70w的数据，3个epoch，一张2080Ti大概200h</mark>
  * <mark style="color:orange;">13B需要18G左右显存（在3090上可以将数据长度开到2048）</mark>
* 推理：一张2080Ti即可（7B）,同时支持多卡推理（差不多均匀负载，某张卡会负载高一点）。
* 我们对纯CPU上推理也进行了支持，详情见[`tools`](https://github.com/Facico/Chinese-Vicuna/blob/master/tools)

### 使用方法

**8bit**

```
bash scripts/finetune.sh
```

这里需要注意的参数如下

* `TOT_CUDA`，填写需要使用的GPU编号，如`TOT_CUDA="0,1,2,3"`
* `PORT`，填写对应的端口
* `DATA_PATH`，填写对应的数据位置，格式为json
* `OUTPUT_PATH`，保存模型的相对路径
* `MODEL_PATH`，上游模型
* `wandb`：这是一个训练可视化工具，脚本中默认没开，可以在脚本中加入"--wandb"来开启

**4bit**

```
bash scripts/finetune_4bit.sh
```

**用于对话式微调**

```
bash scripts/finetune_chat.sh
```

**用于不能开8bit情况/用于fp16的指令式微调**

```
bash scripts/finetune_deepspeed.sh
```

* use\_deepspeed：设置为1表示要用 deepspeed，否则使用fp16 **单卡训练**

```
CUDA_VISIBLE_DEVICES=0 python finetune.py --data_path merge.json --test_size 2000
```

* 这个test\_size不能大于数据大小

**inference并使用gradio生成一个网页（用于指令问答）**

```
bash scripts/generate.sh
```

这里需要注意的参数如下

* BASE\_MODEL，上游模型
* LORA\_PATH，lora模型的checkpoint文件夹
  * 这里要注意的是，lora模型加载的config必须是"adapter\_config.json"，模型名字必须是“adapter\_model.bin”，不过在训练的时候会自动保存为“pytorch\_model.bin”，而"adapter\_config.json"和“adapter\_model.bin”会在全部训练结束之后保存
    * 如果你是在训练的checkpoint中载入的lora模型，代码里会自动帮你把本地的"config-sample/adapter\_config.json"复制到对应目录，并把“pytorch\_model.bin”改名为“adapter\_model.bin”
  * 也可以是任意的huggingface上对应llama 7B的lora模型，如：`Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco`
* USE\_LOCAL，设置为1时会检查本地模型配置
* 使用的时候，"max\_tokens"根据自己电脑的显存来设置，如果生成的内容产生了很多重复信息，可以将"Repetition Penalty"调高

**多轮交互**

由于我们在训练的时候用的基本是指令的prompt，所以闲聊对话能力还比较差，后续将增加这一部分的训练。

```
bash scripts/chat.sh
```

*   使用gradio构造的一个简单的交互界面，可以根据自己的机器设置max\_memory（它会截取历史对话的后面max\_memory部分）


* 这个脚本使用的prompt和generate.sh中使用的不太一样，这个脚本的prompt为对话形式的，如下`The following is a conversation between an AI assistant called Bot and a human user called User.`

同时，为了更好的交互体验，我们自己实现了流式输出（打字机式）交互的chatbot，支持beam search、repetiion penalty的设置，能清空历史记录，选择不同的全局instruction等。

#### **算力配置--**训练的配置是怎么样的？除了7B能训练吗？大于256截断长度能训练吗？最低配置是什么？

训练的硬件要求主要取决于训练序列长度、`mirco batch size`大小，以及训练时间你是否能接受。

* mirco batch size设置小一点，11G的2080Ti也能跑更大长度的
* mirco batch size设置为1，24G的3090Ti可以训练2048长度的
* 4090的算力约为3090两倍，A100-40G的int8算力与4090相近

<table><thead><tr><th width="138"></th><th></th><th width="150"></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th></tr></thead><tbody><tr><td><strong>Model</strong></td><td><strong>GPU</strong></td><td><strong>lora+fp16+512</strong></td><td></td><td><strong>lora+int8+256</strong></td><td></td><td><strong>lora+int8+512</strong></td><td></td><td><strong>lora+int8+2048</strong></td><td></td></tr><tr><td></td><td></td><td>speed</td><td>size</td><td>speed</td><td>size</td><td>speed</td><td>size</td><td>speed</td><td>size</td></tr><tr><td>LLaMA-7B</td><td>2080Ti</td><td></td><td></td><td>0.2h/w</td><td>11G</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>3090</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>4090</td><td>0.3h/w</td><td>20G</td><td></td><td></td><td>0.8h/w</td><td></td><td>3.5h/w</td><td>20G</td></tr><tr><td>LLaMA-13B</td><td>3090</td><td></td><td></td><td>0.9h/w</td><td></td><td></td><td></td><td>7.5h/w</td><td>24G</td></tr><tr><td></td><td>4090</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></tbody></table>

**注意：**

* int8 仅加载模型显存占用（VRAM）硬盘空间大小，比如7B大概8G左右，13B大概14G左右；如果是fp16和fp32则相应乘2和乘4
* 训练的时候，显存占用和训练的速度和序列长度密切相关，比如序列长度256显存占用不超过11G，这个时候可以在2080Ti上微调7B，序列长度如果是2048，则显存占用会骤增到20G，就要上3090或者4090才能微调7B了
* 同理，13B在3090/4090上是可以微调的，2048的时候microbatch降到1也是可以跑的
* 另外，有人发现在A100-40G上增大batch没有明显的提速，这可能是因为int8比较吃算力（比如相同的配置fp16快于int8），算力吃满后增加batch也不能提高吞吐量，另一方面A100-40G的int8算力其实和4090差不多。



### 具体案例

医学问答垂直语料的continue-finetune，参考[Chinese-Vicuna-medical](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/performance-medical.md)



### Reference

LLaMA paper: [https://arxiv.org/abs/2302.13971v1](https://arxiv.org/abs/2302.13971v1)

Self-Instruct paper: [https://arxiv.org/abs/2212.10560](https://arxiv.org/abs/2212.10560)

Data generation: [https://github.com/LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE) and [https://guanaco-model.github.io/](https://guanaco-model.github.io/)

The first work: [https://github.com/tatsu-lab/stanford\_alpaca](https://github.com/tatsu-lab/stanford\_alpaca)

Lora paper：[https://arxiv.org/pdf/2106.09685.pdf](https://arxiv.org/pdf/2106.09685.pdf)



\


