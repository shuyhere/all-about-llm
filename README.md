# all-about-llm
大语言模型训练和服务调研
## LLM
### 1. [百川大模型](https://github.com/baichuan-inc/baichuan-7B)
	
完全支持商用的模型
发布团队：百川智能
家族：llama

注意仓库中的评测结果是基于sft和RLHF 但是目前开源的model是pretrain版本	

预训练模型续写的句子有比较大的问题（可能来自于语料问题？）
{'text': '今天天气是真的有点热,我走在街上的时候,发现了很多人的脸上泛了红色的......\n我走在街上的时候,发现了很多人的脸上泛了红色的,我问老爸:为什么现在有这么多人的脸上都长了斑呢?老爸说:因为天气太热了,脸上没有汗水的滋润,皮肤没有光泽了。你别问了,我现在正在去开会的地方的路上。'}

**7b预计在A100占用显存 ：27689MB  ； 4090占用显存23.6G 推理速度很慢**

####  概述

基于Transformer结构，在大约1.2万亿tokens上训练的70亿参数模型，支持中英双语，上下文窗口长度为**4096**

**针对chat的微调方案（非官方）**：
- MedicalGPT：https://github.com/shibing624/MedicalGPT/blob/main/README.md
- LLaMA-Efficient-Tuning ：
https://github.com/hiyouga/LLaMA-Efficient-Tuning

**问题：评测封闭性不好**？⬇️

	输入：
	在Unix中，passwd命令位于____目录中的。
	输出：
	在Unix中，passwd命令位于____目录中的。 A. /etc/ B. /usr/ C. /bin/ D. /usr/bin/ 答案：A</s>
**分词**
使用SentencePiece中的Byte-PairEncoding(BPE)作为分词算法，并且进行了以下的优化：
* 使用2000万条以中英为主的多语言语料训练分词模型，提升对于中文的压缩率。
* 对于数学领域，参考了LLaMA和Galactica中的方案，对数字的每一位单独分开，避免出现数字不一致的问题，对于提升数学能力有重要帮助。
* 对于罕见字词（如特殊符号等），支持UTF-8characters的byte编码，因此做到未知字词的全覆盖，词表大小达到6.4万。
* 对中文的压缩率比较高（？压缩率咋算的？？？？）

| Model         | baichuan\-7B | LLaMA  | Falcon | mpt\-7B | ChatGLM | moss\-moon\-003 |
|---------------|--------------|--------|--------|---------|---------|-----------------|
| Compress Rate | 0\.737       | 1\.312 | 1\.049 | 1\.206  | 0\.631  | 0\.659          |
| Vocab Size    | 64,000       | 32,000 | 65,024 | 50,254  | 130,344 | 106,029         |

#### 数据
![数据处理流程][def2]

#### 模型结构
整体模型基于标准的 Transformer 结构，采用了和 LLaMA 一样的模型设计
![模型结构][def]
- **位置编码** rotary-embedding 是现阶段被大多模型采用的位置编码方案，具有更好的外延效果。虽然训练过程中最大长度为4096，但是实际测试中模型可以很好的扩展到 5000 tokens 上，如下图：
  ![Alt text][def3]
- **具体参数**
	| **Hyperparameter** | **Value**  |
	|--------------------|------------|
	| n_parameters       | 7000559616 |
	| n_layers           | 32         |
	| n_heads            | 32         |
	| d_model            | 4096       |
	| vocab size         | 64000      |
	| sequence length    | 4096       |


  

- 激活层：SwiGLU, Feedforward 变化为(8/3)倍的隐含层大小，即11008
- Layer-Normalization: 基于 RMSNorm 的 Pre-Normalization
#### 训练算力
千卡A800机器上达到了7B模型182Tflops的吞吐，GPU峰值算力利用率高达58.3% 。训练稳定性和吞吐
#### 模型推理
[huggingface](https://huggingface.co/baichuan-inc/baichuan-7B)


### 2. [Aquila 悟道天鹰系列商用开源模型](https://github.com/FlagAI-Open/FlagAI/blob/master/examples/Aquila/README.md)

发布团队：智源社区
要用flagAI框架（这个框架源码&依赖的兼容性还有一些错误）
关注issue：https://github.com/FlagAI-Open/FlagAI/issues/334
这个问题官方正在修复
#### 概述
**官方简介**
Aquila语言大模型在技术上继承了GPT-3、LLaMA等的架构设计优点，替换了一批更高效的底层算子实现、重新设计实现了中英双语的tokenizer，升级了BMTrain并行训练方法，在Aquila的训练过程中实现了比Magtron+DeepSpeed zero-2将近８倍的训练效率。Aquila语言大模型是在中英文高质量语料基础上从０开始训练的，通过数据质量的控制、多种训练的优化方法，实现在更小的数据集、更短的训练时间，获得比其它开源模型更优的性能。也是首个支持中英双语知识、支持商用许可协议、符合国内数据合规需要的大规模开源语言模型。

**最低硬件需求**： **运行Aquila-7B系列需要内存30G, 显存18G，生成最大长度 2048 tokens。**

Aquila模型所采用的tokenizer是由从头开始训练的，支持中英双语。我们在处理英文、中文以及代码数据时，采用了不同的分词器对一万个样本进行了抽取。Aquila tokenizer与其他tokenizer的参数对比见下表:
| **模型/Model** | **词表大小/Vocab size** | **说明/Note** | **英文平均tokens量/Avg tokens \(English \)** | **中文平均tokens量/Avg tokens \(Chinesse \)** | **代码平均tokens量/Avg tokens \(code\)** |
|--------------|---------------------|-------------|---------------------------------------|----------------------------------------|------------------------------------|
| GPT2         | 50527               | bpe         | 1717                                  | 1764                                   | 2323                               |
| LlaMA        | 32000               | sp\(bpe\)   | 1805                                  | 1257                                   | 1970                               |
| Aquila       | 100000              | bpe         | 1575                                  | 477                                    | 1679                               |

#### 模型下载
- 脚本下载
-  FlagOpen 模型仓库下载 https://model.baai.ac.cn/models

其他信息等待六月底官方技术报告



### 3. [Chinese-Vicuna: A Chinese Instruction-following LLaMA-based Model —— 一个中文低资源的llama+lora方案](https://github.com/Facico/Chinese-Vicuna)

这种方案可能会降低模型的泛化能力

git仓库提供了一个针对llama系列工程很好的QA文档：https://github.com/Facico/Chinese-Vicuna/blob/master/docs/notes.md

有关技术
- LLaMA paper: https://arxiv.org/abs/2302.13971v1
- Self-Instruct paper: https://arxiv.org/abs/2212.10560
- data generation: https://github.com/LianjiaTech/BELLE and https://guanaco-model.github.io/
- the first work: https://github.com/tatsu-lab/stanford_alpaca
- Lora paper：https://arxiv.org/pdf/2106.09685.pdf
#### 概述
**特点**：llama+中文+低资源+垂料训练的方案 
  基于LLaMA+instruction数据构建一个中文的羊驼模型，并帮助大家能快速学会使用引入自己的数据，并训练出属于自己的小羊驼（Vicuna）
  类似于stable diffusion模型的爆火，出现了像civitai等平台，由一个基础的模型+各种LORA模型的开源社区。这个项目就是去训练这个LORA
**Lora训练方式**：
- 代码基于alpaca-lora开发，https://github.com/tloen/alpaca-lora
- 这是一套比较简单的代码，基本思路就是用PEFT的lora接口+transformer的trainer+instruction的数据配置
**训练算力配置：**（来源为项目github）
	4090的算力约为3090两倍，A100-40G的int8算力与4090相近
	| **Model** | **GPU** | **lora+fp16+512** |  | **lora+int8+256** |  | **lora+int8+512** |  | **lora+int8+2048** | |
	|-----------|---------|--------------------|------|--------------------|------|--------------------|------|---------------------|------|
	|           |         | speed              | size | speed              | size | speed              | size | speed               | size |
	| LLaMA-7B  | 2080Ti  |                    |      | 0.2h/w            | 11G  |                    |      |                     |      |
	|           | 3090    |                    |      |                    |      |                    |      |                     |      |
	|           | 4090    | 0.3h/w            | 20G  |                    |      | 0.8h/w            |      | 3.5h/w              | 20G  |
	| LLaMA-13B | 3090    |                    |      | 0.9h/w            |      |                    |      | 7.5h/w              | 24G  |
	|           | 4090    |


**注意：**
- int8 仅加载模型显存占用（VRAM）$ \approx $ 硬盘空间大小，比如7B大概8G左右，13B大概14G左右；如果是fp16和fp32则相应乘2和乘4
- 训练的时候，显存占用和训练的速度和序列长度密切相关，比如序列长度256显存占用不超过11G，这个时候可以在2080Ti上微调7B，序列长度如果是2048，则显存占用会骤增到20G，就要上3090或者4090才能微调7B了；
- 同理，13B在3090/4090上是可以微调的，2048的时候microbatch降到1也是可以跑的
- 另外，有人发现在A100-40G上增大batch没有明显的提速，这可能是因为int8比较吃算力（比如相同的配置fp16快于int8），算力吃满后增加batch也不能提高吞吐量，另一方面A100-40G的int8算力其实和4090差不多。
#### 数据
利用了目前几份高质量的开源数据（这些数据很多都像alpaca那样，使用chatgpt的接口，生成高质量的instruction数据。）
{"instruction": "用一句话描述地球为什么是独一无二的。\\n\n", "input": "", "output": "地球上有适宜生命存在的条件和多样化的生命形式。"}
- Belle https://github.com/LianjiaTech/BELLE
- guanaco https://huggingface.co/datasets/JosephusCheung/GuanacoDataset
数据下载链接
- 有更好对话能力的chatv1的微调数据：https://huggingface.co/datasets/Chinese-Vicuna/instruct_chat_50k.jsonl ：由3万条sharegpt中文数据和2万条alpaca-instruction-Chinese-dataset数据组成
- 链接: https://pan.baidu.com/s/1WSxuhSAotl14ifaAiz5eKw?pwd=b4kb 提取码: b4kb
- 链接: https://drive.google.com/file/d/1tzXVhS74m-EtoFot7hEc005LDeZGPit_/view?usp=sharing
- 链接: https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0 （694K rows  409MB）
#### 模型
- 上游模型：LLAMA 7B
- 项目提供了lora模型
  - 加载方式参考generate.py
    - Chinese-Vicuna/Chinese-Vicuna-lora-7b-belle-and-guanaco
    - Chinese-Vicuna/Chinese-Vicuna-lora-13b-belle-and-guanaco
  - 模型使用的是8bit+lora+256 tokens
  - 更多模型：https://huggingface.co/Chinese-Vicuna

#### 4.  [Chinese-LLaMA-Alpaca 开源中文LLaMA模型和指令精调的Alpaca大模型](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

chinese-vicuna中提到这个项目：**做了词表扩充但是效果不及没有扩充词表的fastchat-vicuna**

这个项目给了一个扩充词表的思路&开源了很多有用的工具和代码;项目wiki有很多干货;训练代价比较大

#### 概述

**技术报告**：[Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca](https://arxiv.org/abs/2304.08177)

- 🚀 针对原版LLaMA模型扩充了中文词表，提升了中文编解码效率
- 🚀 开源了使用中文文本数据预训练的中文LLaMA以及经过指令精调的中文Alpaca
- 🚀 开源了预训练脚本、指令精调脚本，用户可根据需要进一步训练模型
- 🚀 快速使用笔记本电脑（个人PC）的CPU/GPU本地量化和部署体验大模型
- 🚀 支持🤗transformers, llama.cpp, text-generation-webui, LlamaChat, LangChain, privateGPT等生态
- 目前已开源的模型版本：7B（基础版、Plus版）、13B（基础版、Plus版）、33B（基础版） （模型大小相同 plus版本使用的数据更多）
  
**训练配置：**
以下是训练基础版7B模型的训练配置。更多详情请参考技术报告。

| **实验设置**                   | **预训练-第一阶段**     | **预训练-第二阶段**     | **指令精调**          |
|----------------------------|-------------------|-------------------|-------------------|
| Batch Size                 | 1024              | 1024              | 512               |
| Initial Learning Rate      | 2.00E-04        | 1.00E-04        | 1.00E-04        |
| Training Steps             | 3K                | 6K                | 6K-10K           |
| Max Length                 | 512               | 512               | 512               |
| Trainable Parameters (%) | 2.97%            | 6.06%            | 6.22%            |
| Training Device            | 8 × A100          | 16 × A100         | 16 × A100         |
| Distributed Training       | DeepSpeed Zero-2 | DeepSpeed Zero-2 | DeepSpeed Zero-2 |

**训练细节**
- 在通用中文语料上训练了基于sentencepiece的20K中文词表并与原版LLaMA模型的32K词表进行合并--最终词表大小49953&49954
- 更多细节见：https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E8%AE%AD%E7%BB%83%E7%BB%86%E8%8A%82


#### 模型
发布的是LoRA权重，需要搭配[原版LLaMA模型](https://github.com/facebookresearch/llama)，然后[合并模型](https://github.com/ymcui/Chinese-LLaMA-Alpaca#%E5%90%88%E5%B9%B6%E6%A8%A1%E5%9E%8B)。
下载后务必检查压缩包中模型文件的SHA256是否一致，请查看[SHA256.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/SHA256.md)。

**模型对比**

| **对比项**                  | **中文LLaMA**                 | **中文Alpaca**                            |
|--------------------------|-----------------------------|-----------------------------------------|
| 训练方式                     | 传统CLM                       | 指令精调                                    |
| 训练语料                     | 无标注通用语料                     | 有标注指令数据                                 |
| 词表大小              | 49953                       | 49954=49953\+1（pad token）               |
| 输入模板                     | 不需要                         | 需要符合模板要求                         |
| 适用场景 ✔️                  | 文本续写：给定上文内容，让模型继续写下去，生成下文   | "1、指令理解（问答、写作、建议等）                      |
|多轮上下文理解（聊天等）          |
| 不适用场景 ❌                  | 指令理解 、多轮聊天等                 | 文本无限制自由生成                               |
| llama\.cpp               | 使用\-p参数指定上文                 | 使用\-ins参数启动指令理解\+聊天模式                   |
| text\-generation\-webui  | 不适合chat模式                   | 使用\-\-cpu可在无显卡形式下运行，若生成内容不满意，建议修改prompt |
| LlamaChat                | 加载模型时选择"LLaMA"              | 加载模型时选择"Alpaca"                         |
| HF推理代码                   | 无需添加额外启动参数                  | 启动时添加参数 \-\-with\_prompt                |
| web\-demo代码              | 不适用                         | 直接提供Alpaca模型位置即可；支持多轮对话                 |
| LangChain示例 / privateGPT | 不适用                         | 直接提供Alpaca模型位置即可                        |
| 已知问题                     | 如果不控制终止，则会一直写下去，直到达到输出长度上限。 | 目前版本模型生成的文本长度相对短一些，比较惜字如金。可在指令中要求详细回答。  |

**Lora下载：**
| **模型名称**                     | **训练数据** | **重构模型**     | **大小** | **LoRA下载** |
|------------------------------|----------|-------------------|-------------|-----------------|
| Chinese\-LLaMA\-7B           | 通用20G    | 原版LLaMA\-7B       | 770M        | [百度网盘](https://pan.baidu.com/s/1xV1UXjh1EPrPtXg6WyG7XQ?pwd=923e) [googledrive](https://drive.google.com/file/d/1JvFhBpekYiueWiUL3AF1TtaWDb3clY5D/view?usp=sharing)    |          |
| Chinese\-LLaMA\-Plus\-7B ⭐️  | 通用120G   | 原版LLaMA\-7B       | 790M        |[百度网盘](https://pan.baidu.com/s/12tjjxmDWwLBM8Tj_7FAjHg?pwd=32hc) [googledrive](https://drive.google.com/file/d/1EDcTmq6tDmRxqarpapdyDGBE9opY0zrB/view?usp=share_link)     |         |
| Chinese\-LLaMA\-13B          | 通用20G    | 原版LLaMA\-13B      | 1\.0G       | [百度网盘](https://pan.baidu.com/s/1wYoSF58SnU9k0Lndd5VEYg?pwd=mm8i) [googledrive](https://drive.google.com/file/d/1gzMc0xMCpXsXmU1uxFlgQ8VRnWNtDjD8/view?usp=share_link)       |          |
| Chinese\-LLaMA\-Plus\-13B ⭐️ | 通用120G   | 原版LLaMA\-13B      | 1\.0G       | [百度网盘](https://pan.baidu.com/s/1Mew4EjBlejWBBB6_WW6vig?pwd=mf5w) [googledrive](https://drive.google.com/file/d/1CcLJvY7XsFAOjfSIqCpDI7jf3EEPDcEF/view?usp=share_link)       |              |
| Chinese\-LLaMA\-33B          | 通用20G    | 原版LLaMA\-33B\ | 2\.7G       |  [百度网盘](https://pan.baidu.com/s/1fey7lGMMw3GT982l8uJYMg?pwd=2f2s) [googledrive](https://drive.google.com/file/d/1YeSgnZWaRkKdmYa-JHiIlcvqhrDd4-Y4/view?usp=share_link)          |

同时：可以在🤗Model Hub下载以上所有模型，并且使用transformers和PEFT调用中文LLaMA或Alpaca LoRA模型。详细见项目git仓库

#### 量化推理和部署
[参考](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E4%B8%8E%E9%83%A8%E7%BD%B2)

#### 数据

**预训练语料**
在预训练阶段，使用约20G左右的通用中文语料（与中文BERT-wwm、MacBERT、LERT、PERT中使用的语料一致）在原版LLaMA权重的基础上进一步进行预训练

**指令精调语料**
| **数据**              | **量级** | **来源** | **说明**                         |
|---------------------|--------|--------|--------------------------------|
| 中英翻译数据              | 500K   | [外部链接](https://github.com/brightmart/nlp_chinese_corpus#5%E7%BF%BB%E8%AF%91%E8%AF%AD%E6%96%99translation2019zh)   | 在原数据集的基础上进行了采样\+规则筛选           |
| pCLUE数据             | 300K   | [外部链接](https://github.com/CLUEbenchmark/pCLUE)   | 在原数据集的基础上进行了采样\+规则筛选           |
| Alpaca数据（英）         | 50K    | [外部链接](htthttps://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/dataps://github.com/tatsu-lab/stanford_alpaca)  | 斯坦福原版Alpaca训练数据                |
| Alpaca数据（中）         | 50K    | [外部链接]()  | 本项目使用ChatGPT接口将英文版翻译为中文（筛掉一部分） |
| Self\-instruction数据 | 1~2M   | （暂无）   | 本项目使用ChatGPT接口进行爬取，具体见以下脚本描述   |

本项目提供了一个动态生成不同领域和指令类型的prompt爬取脚本 `script/crawl_prompt.py`

### 5.  [BELLE: Be Everyone's Large Language Model Engine](https://github.com/LianjiaTech/BELLE)

相比如何做好大语言模型的预训练，BELLE更关注如何在开源预训练大语言模型的基础上，帮助每一个人都能够得到一个属于自己的、效果尽可能好的具有指令表现能力的语言模型，降低大语言模型、特别是中文大语言模型的研究和应用门槛.

### 6.[Tigerbot](https://github.com/TigerResearch/TigerBot)
  - [ ] 为什么tigerbot使用100w数据指令微调的结果比不上vicuna使用50k数据微调的结果，是pretraining模型有问题还是bloom底座就是不行呢（用basemodel测试一下）
  
模型对标instuctGPT-6B 目前7b模型的效果一般 可能训练过程有很多脏数据 比较明显的问题是会输出各种多余的符号

demo网页：https://tigerbot.com/chat

#### 概述

**发布机构**：虎博科技
**家族**：bloom  
关于为什么使用bloom基座而不是更火3的llama：1 bloom基座可以商用 / 2 项目开始时间早于llama
**成员**：TigerBot-7B, TigerBot-7B-base，TigerBot-180B (research version)
**功能**：提供了三种 API，包括 Chat-API，Plug-ins，**Fine-Tunes**。
**数据**：预训练 100G，从 2TB 过滤后的数据中经过去噪去重清洗而得；监督微调 1G 或 100 万条数据，按比例涵盖用户指令常见的 10 大类 120 小类任务

**官方介绍:**
在 BLOOM 基础上，在模型架构和算法上做了如下优化：
- 指令完成监督微调的创新算法以获得更好的可学习型(learnability)，
- 运用 ensemble 和 probabilistic modeling 的方法实现更可控的事实性(factuality)和创造性(generativeness)，
- 在并行训练上，突破了 deep-speed 等主流框架中若干内存和通信问题，使得在千卡环境下数月无间断（负责人说是改了deepseed源码实现）
- 对中文语言的更不规则的分布，从 tokenizer 到训练算法上做了更适合的算法优化。
量化：使用GPTQ算法和GPTQ-for-LLaMa实现量化：

**训练算力**：
7b模型  单卡a100 40G就可以训练
180b模型  
**推理算力**：
量化前：`tigerbot-7b-sft` 推理可在1张RXT3090上进行；`tigerbot-180b-sft` 推理可在5张A100(80G)上进行    
**量化后**：`tigerbot-7b-sft-4bit-128g` 推理可在一张RTX3090上进行 `tigerbot-180b-research-4bit-128g` 推理可在两张A100(80G)上进行

#### 数据

**预训练数据**
基于 GPT3 的 pretrain 的数据分布，采集中文书籍，互联网，和百科类数据，并通过数据源质量分过滤和 tf-idf soft deduping，从 20TB 数据过滤到 2TB，保持语言和类目的比例，并在此基础上随机抽样 100G 数据开源
开源数据集
- [中文开源预训练集 - 55G，包含中文书籍、中文互联网、中文百科](https://huggingface.co/datasets/TigerResearch/pretrain_zh)
- [英文开源预训练集 - 51G，包含英文书籍、英文互联网、英文百科](https://huggingface.co/datasets/TigerResearch/pretrain_en)
  
**微调数据**
数据下载链接和清洗规则见项目github

开源数据集
- 指令数据集, 当前开源 120W 问答对，磁盘空间 1.1G （数据集开放到 huggingface）
- 领域数据开放金融、法律、百科相关领域数据，作为 rethink 外部数据
  

#### 模型下载
| **Tigerbot\-7B**              | **Bits** | **memory\(GB\)** |
|-------------------------------|----------|------------------|
| tigerbot\-7b\-base            | 16       | 17\.2            |
| tigerbot\-7b\-sft             | 16       | 17\.2            |
| tigerbot\-7b\-sft\-4bit\-128g | 4        | 8\.5             |

| **Tigerbot\-180B\-Research**    | **Bits** | **memory\(GB\)** |
|---------------------------------|----------|------------------|
| tigerbot\-180b\-sft             | 16       | 347\.6           |
| tigerbot\-180b\-sft\-4bit\-128g | 4        | 108\.5           |

### 7. [Fastchat](https://github.com/lm-sys/FastChat)

FastChat 是一个开放平台，用于训练、服务和评估基于大型语言模型的聊天机器人。核心功能包括：

最先进模型（例如，Vicuna、FastChat-T5）的权重、训练代码和评估代码。
具有 Web UI 和与 OpenAI 兼容的 RESTful API 的分布式多模型服务系统。


### 8. Vicuna

用 ShareGPT 收集的对话数据微调 LLaMA实现   用GPT-4评价效果
不擅长数学推理、编码任务，语言逻辑上整体是英文逻辑（比如中文任务会导致状语后置&夹杂英文单词等）

![Alt text][def4]
使用openai [moderation](https://platform.openai.com/docs/guides/moderation/overview) api来进行审核 

#### 概述


**数据**：70K 对话对
**训练**：增强了 Alpaca 提供的训练脚本，以更好地处理多轮对话和长序列。一天内在 8 个 A100 GPU 上使用 PyTorch FSDP 完成
**算力需求**：训练 ：8 个 A100 GPU

官方[blog](https://lmsys.org/blog/2023-03-30-vicuna/#how-good-is-vicuna)提供了一个对比表格：
 **Model Name**          | **LLaMA**                                | **Alpaca**                                           | **Vicuna**                                 | **Bard/ChatGPT** |
|-------------------------|------------------------------------------|------------------------------------------------------|--------------------------------------------|------------------|
| Dataset                 | Publicly available datasets\(1T token\) | Self\-instruct from davinci\-003 API\(52K samples\) | User\-shared conversations\(70K samples\) | N/A              |
| Training code           | N/A                                      | Available                                            | Available                                  | N/A              |
| Evaluation metrics      | Academic benchmark                       | Author evaluation                                    | GPT\-4 assessment                          | Mixed            |
| Training cost(7B)   | 82K GPU\-hours                           | \$ 500 (data) + $100 (training)                   | $140(training)                          | N/A              |
| Training cost (13B) | 135K GPU\-hours                          | N/A                                                  | $300 (training)                          | N/A              |

#### 数据
从 ShareGPT.com 使用公共 API 收集的大约 70K 用户共享对话
为了确保数据质量，将 HTML 转换回 markdown 并过滤掉一些不合适或低质量的样本，将冗长的对话分成适合模型最大上下文长度的较小片段


#### 训练过程 
训练脚本在 Stanford’s alpaca 的基础上做了如下改进：
- 内存优化：为了使 Vicuna 能够理解长上下文，将最大上下文长度从羊驼中的 512 扩展到 2048，这大大增加了 GPU 内存需求。通过利用梯度检查点和闪存注意力来解决内存压力。 
- 多轮对话：调整训练损失以考虑多轮对话，并仅根据聊天机器人的输出计算微调损失。 
- 通过 Spot Instance降低成本：40 倍大的数据集和 4 倍的训练序列长度对训练费用提出了相当大的挑战。 vicuna使用 **SkyPilot managed spot**来降低成本（利用更便宜的点实例以及自动恢复抢占和自动区域切换）。 该解决方案将 7B 模型的训练成本从 500 美元削减至 140 美元左右，将 13B 模型的训练成本从 1000 美元左右削减至 300 美元。 （AlpaServe ）

### 模型 
**地址**: https://huggingface.co/lmsys

还是delta的格式发布权重 类似这样转换 ↓
需要资源：30GB of CPU RAM（7b） 60GB of CPU RAM（13b）提供了 Low CPU Memory `Conversion 方式：--low-cpu-mem 
python3 -m fastchat.model.apply_delta \
    --base-model-path /path/to/llama-7b \
    --target-model-path /path/to/output/vicuna-7b \
    --delta-path lmsys/vicuna-7b-delta-v1.1  
--low-cpu-mem`

### 9. [Koala: A Dialogue Model for Academic Research 考拉对话模型](https://bair.berkeley.edu/blog/2023/04/03/koala/)

![Alt text](./figure/image3.png)

### 10.[Firefly（流萤）: 中文对话式大语言模型](https://github.com/yangjianxin1/Firefly)

**家族** : bloom 
- 开源QLoRA的训练代码，使用一张显卡对bloom-7b1进行微调，开源firefly-7b1-qlora-v0.1模型 。
https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M
本数据应用于项目：Firefly（流萤）: 中文对话式大语言模型 ，训练后得到的模型[firefly-1b4](https://huggingface.co/YeungNLP/firefly-bloom-1b4-sft)

### 11. Stanford Alpaca: An Instruction-following LLaMA Model 

### 12. [GPT4all](https://github.com/nomic-ai/gpt4all)

## 多模态 Model & 多模态任务实现方案
#### 1. [I-JEPA (the Image-based Joint-Embedding Predictive Architecture)](https://arxiv.org/pdf/2301.08243.pdf)
meta开源
首个世界背景知识学习模型 
图像补全任务 https://github.com/facebookresearch/ijepa

### 2. Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding
https://arxiv.org/abs/2306.02858

### 3. MM-React 

### 4. Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models



## 大模型有关项目实现 工具&参考项目

### 优化方向
1. 优化 text_split 算法，使匹配出的结果作为上下文时能够提供更合理的推理/回答依据；
2. 优化 embedding 模型，提升语义向量化的效果，使得语义匹配过程中能够匹配出最满足要求的文本段落作为上下文；
3. 优化 LLM 模型，使得给定提问相同情况下，得到更理想的推理/回答结果。

### 数据准备
- 分词工具 GitHub - google/sentencepiece: Unsupervised text tokenizer for Neural Network-based text generation. 在chinese-llama&aplace项目中用于词表扩充
- 网站爬虫工具 https://apify.com/apify/website-content-crawler
- PDF等文本解析组件：
  1、PDFminer
  PDFMiner是一个Python的PDF解析器，可以从PDF文档中提取信息。与其他PDF相关的工具不同，它侧重的是获取和分析文本数据。PDFMiner允许获取某一页中文本的准确位置和一些诸如字体、行数的信息。它包括一个PDF转换器，可以把PDF文件转换成HTML等格式。还有一个扩展的PDF解析器，可以用于除文本分析以外的其他用途。
  PDFMiner内置两个工具：pdf2txt.py和dumppdf.py：
  pdf2txt.py从PDF文件中提取所有文本内容。但不能识别画成图片的文本，这需要特征识别。对于加密的PDF你需要提供一个密码才能解析，对于没有提取权限的PDF文档你得不到任何文本。
  地址：https://pdfminersix.readthedocs.io
  2、pdfplumber
  pdfplumber库按页处理 pdf ，获取页面文字，提取表格等操作。
  地址：https://github.com/jsvine/pdfplumber
  3、pypdf2
  PyPDF2是一个纯Python PDF库，可以读取文档信息（标题，作者等）、写入、分割、合并PDF文档，它还可以对pdf文档进行添加水印、加密解密等。
  地址：https://pythonhosted.org/PyPDF2
  4、pymupdf
  PyMuPDF是支持MuPDF的Python绑定。
  使用PyMuPDF，可以访问扩展名为“.pdf”、“.xps”、“.oxps”、“.cbz”、“.fb2”或“.epub”。此外，大约10种流行的图像格式也可以像文档一样处理:“.png”，“.jpg”，“.bmp”，“.tiff”等。
  地址：https://pypi.org/project/PyMuPDF/
  5、ppstructure
  PP-StructureV2支持对图片/pdf形式的文档进行版面分析，可以划分文字、标题、表格、图片、公式等区域；支持通用的中英文表格检测任务；支持表格区域进行结构化识别，最终结果输出Excel文件；
  PP-Structure是PaddleOCR团队自研的智能文档分析系统，旨在帮助开发者更好的完成版面分析、表格识别等文档理解相关任务。
  版面分析任务中，图像首先经过版面分析模型，将图像划分为文本、表格、图像等不同区域，随后对这些区域分别进行识别，如，将表格区域送入表格识别模块进行结构化识别，将文本区域送入OCR引擎进行文字识别，最后使用版面恢复模块将其恢复为与原始图像布局一致的word或者pdf格式的文件。
  地址：https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/ppstructure

- 文本向量化组件
  1、text2vec
  实现了Word2Vec、RankBM25、BERT、Sentence-BERT、CoSENT等多种文本表征、文本相似度计算模型，并在文本语义匹配（相似度计算）任务上比较了各模型的效果。
  地址：https://github.com/shibing624/text2vec
     2、SGPT
  SGPT：GPT Sentence Embeddings for Semantic Search，是一个使用GPT架构生成embedding的方法，与BERT模式不同。
  地址：https://arxiv.org/abs/2202.08904
   3、M3E
  Moka Massive Mixed Embedding的缩写，由MokaAI训练，训练脚本使用 uniem，评测BenchMark使用MTEB-zh，通过千万级 (2200w+) 的中文句对数据集进行训练。


### 数据处理
- Vectara  数据筛选和匹配工具 github:https://github.com/vectara/vectara-answer
效果： 匹配效果针对于中文不太好，但是识别文字的能力，界面的设置，文字的章节识别等很好，数据库式管理，适合批量处理文档
	![Alt text](./figure/image4.png)

### 模型下载&转换，训练&微调
- Training Open Instruction-following Language Models https://github.com/allenai/open-instruct
- 各种模型的低资源量化和部署 https://github.com/jianzhnie/Efficient-Tuning-LLMs
- Chinese-vicuna 项目提供  Chinese-Vicuna/readme_zh.md at master · Facico/Chinese-Vicuna
- Chinese-Llama-Aplace项目提供:
  
| **方式** | **适用场景**                            | **教程** |
|--------|-------------------------------------|--------|
| 在线转换   | Colab用户可利用本项目提供的notebook进行在线转换并量化模型 | [链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E5%9C%A8%E7%BA%BF%E6%A8%A1%E5%9E%8B%E5%90%88%E5%B9%B6%E4%B8%8E%E8%BD%AC%E6%8D%A2)     |
| 手动转换   | 离线方式转换，生成不同格式的模型，以便进行量化或进一步精调       | [链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E6%89%8B%E5%8A%A8%E6%A8%A1%E5%9E%8B%E5%90%88%E5%B9%B6%E4%B8%8E%E8%BD%AC%E6%8D%A2)     |
- [MeZO: Fine-Tuning Language Models with Just Forward Passes](https://security.feishu.cn/link/safety?target=https%3A%2F%2Farxiv.org%2Fpdf%2F2305.17333.pdf&scene=ccm&logParams=%7B%22location%22%3A%22ccm_default%22%7D&lang=zh)
  内存高效的零阶优化器，实现与推理阶段相同的内存占用大小
- Fairseq https://github.com/facebookresearch/fairseq
- Deepseed

### 模型量化推理和部署

- 模型量化 https://github.com/qwopqwop200/GPTQ-for-LLaMa
- Chinese-Llama-Aplace项目整理   具体内容请参考本项目 >>> 📚 https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E4%B8%8E%E9%83%A8%E7%BD%B2


| **推理和部署方式**             | **特点**                            | **平台** | **CPU** | **GPU** | **量化加载** | **图形界面** | **教程** |
|-------------------------|-----------------------------------|--------|---------|---------|----------|----------|--------|
| [llama\.cpp](https://github.com/ggerganov/llama.cpp)              | 丰富的量化选项和高效本地推理                    | 通用     | ✅       | ✅       | ✅        | ❌        | [链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/llama.cpp%E9%87%8F%E5%8C%96%E9%83%A8%E7%BD%B2)     |
| 🤗[Transformers](https://github.com/huggingface/transformers)         | 原生transformers推理接口                | 通用     | ✅       | ✅       | ✅        | ✅        | [链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E4%BD%BF%E7%94%A8Transformers%E6%8E%A8%E7%90%86)     |
| [text\-generation\-webui](https://github.com/oobabooga/text-generation-webui) | 前端Web UI界面的部署方式                   | 通用     | ✅       | ✅       | ✅        | ✅        | [链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E4%BD%BF%E7%94%A8text-generation-webui%E6%90%AD%E5%BB%BA%E7%95%8C%E9%9D%A2)     |
| [LlamaChat](https://github.com/alexrozanski/LlamaChat)               | macOS下的图形交互界面（需搭配llama\.cpp模型）    | MacOS  | ✅       | ❌       | ✅        | ✅        | [链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E4%BD%BF%E7%94%A8LlamaChat%E5%9B%BE%E5%BD%A2%E7%95%8C%E9%9D%A2%EF%BC%88macOS%EF%BC%89)     |
| [LangChain](https://github.com/hwchase17/langchain)               | LLM应用开发框架，适用于进行二次开发               | 通用     | ✅†      | ✅       | ✅†       | ❌        | [链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E4%B8%8ELangChain%E8%BF%9B%E8%A1%8C%E9%9B%86%E6%88%90)     |
| [privateGPT](https://github.com/imartinez/privateGPT)              | 基于LangChain的多文档本地问答框架             | 通用     | ✅       | ✅       | ✅        | ❌        | [链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E4%BD%BF%E7%94%A8privateGPT%E8%BF%9B%E8%A1%8C%E5%A4%9A%E6%96%87%E6%A1%A3%E9%97%AE%E7%AD%94)     |
| [Colab Gradio Dem](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/notebooks/gradio_web_demo.ipynb)       | 在Colab中启动基于Gradio的交互式Web服务，体验模型效果 | 通用     | ✅       | ✅       | ✅        | ❌        | [链接](https://colab.research.google.com/github/ymcui/Chinese-LLaMA-Alpaca/blob/main/notebooks/gradio_web_demo.ipynb)     |


### 框架
- Langchain
- llama-index
- YuLan-IR/RETA-LLM at main · RUC-GSAI/YuLan-IR  YuLan-RETA-LLM：在大语言模型中使用检索
  

### 效果评估
* Evaluating instruction following on more user-oriented data
* C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models  

### 常见bug解决
在python=3.9环境中运行langchian时候遇到not a class错误：

`pip install typing-inspect==0.8.0 typing_extensions==4.5.0`


## 参考资料

### Paper
- A Survey of Large Language Models
- Transformer models: an introduction and catalog — 2023 Edition

### 工具网址
- https://civitai.com/
  
### 一些值得关注的issues
- Retrieval-Augmented Generation 检错增强技术 
RAG——使用检索增强生成构建特定行业的大型语言模型
- 是否要进行词表扩充  [常见问题](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98#%E9%97%AE%E9%A2%984%E4%B8%BA%E4%BB%80%E4%B9%88%E8%A6%81%E6%89%A9%E5%85%85%E8%AF%8D%E8%A1%A8%E7%9B%B4%E6%8E%A5%E5%9C%A8%E5%8E%9F%E7%89%88llama%E4%B8%8A%E7%94%A8%E4%B8%AD%E6%96%87%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8D%E8%A1%8C%E5%90%97)
“LLaMA词表中仅包含很少的中文字符，所以在切词时会把中文切地更碎，需要多个byte token才能拼成一个完整的汉字，进而导致信息密度降低。比如，在扩展词表后的模型中，单个汉字倾向于被切成1个token，而在原版LLaMA中可能就需要2-3个才能组合成一个汉字，显著降低编解码的效率。”  （这可能就是在用vicuna的时候虽然是2048tokns但是实际用起来上下文很短的原因）

- “垂直领域的数据量过大会导致模型失去泛用能力，甚至失去语言能力，即说人话的能力”：https://github.com/Facico/Chinese-Vicuna/issues/68
  
- 同上 ptuning lora等小参数微调方法容易产生的灾难性遗忘问题
“最好能够在微调语料中也加入通用学习语料一起微调，避免产生对微调语料极大的偏向，在instruct gpt论文中也提到在强化学习ppo的时候模型也会很容易对于ppo数据拟合，降低模型通用自然语言任务能力，所以在ppo loss中加入了SFT梯度和预训练梯度来缓解这种遗忘问题。”

- 优化方向 https://github.com/imClumsyPanda/langchain-ChatGLM/issues/14
  
- fintuning数据准备问题  https://github.com/THUDM/ChatGLM-6B/issues/364
  doc2query/msmarco-chinese-mt5-base-v1， 根据doc生成问题

### LLM成长路线图

示例：https://github.com/LAION-AI/Open-Assistant#the-plan

[def]: ./figure/image0.png
[def2]: ./figure/image.png
[def3]: ./figure/image1.png
[def4]: ./figure/image2.png