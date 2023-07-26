---
description: 伦敦大学、MetaAI、StabilityAI联合发布70页综述，盘点大模型的16大挑战
---

# 👏 Challenges and Applications of Large Language Models

论文链接：[https://arxiv.org/abs/2307.10169](https://arxiv.org/abs/2307.10169)

本文关注两个问题：

(1) Challenges: What problems remain unresolved?  -- 从设计、行为、科学三个方面展开

(2) Applications: Where are LLMs currently being applied, and how are the challenges constraining them? -- 介绍了聊天机器人、计算生物学、计算机编程、创造性工作、知识工作、法律、医学、推理、机器人和社会科学等领域。

## Challenges

<figure><img src="../../.gitbook/assets/image (3).png" alt="" width="563"><figcaption></figcaption></figure>

### Challenge 1：Unfathomable Datasets 深不可测的数据集：The size of modern pre-training datasets renders it impractical for any individual to read or conduct quality assessments on the encompassed documents thoroughly. 数据规模太大，任何个人都无法彻底阅读所包含的文档或对其进行质量评估

扩大预训练数据量一直是为LLM配备通用功能的主要驱动力之一。 预训练数据集的大小很快就超过了大多数人类团队可以手动进行质量检查的文档数量。 相反，大多数数据收集过程依赖于有关数据源和过滤的启发式（heuristics）方法。

这一部分探讨了这些启发法的不利后果，以及许多模型实践者对其模型训练所用的数据只有模糊理解的现实。&#x20;

#### **Near-Duplicates**&#x20;

**近似重复数据**会影响模型性能，而过滤这些数据很困难。现有方法: MinHash; NearDup;  SemDeDup; Kaddour 提出的 [The MiniPile Challenge for Data-Efficient Language Models](https://arxiv.org/abs/2304.0844)...

#### **Benchmark Data Contamination**

当训练数据集包含来自或类似于评估测试集的数据时，就会发生**基准数据污染**。这可能会导致性能指标夸大，因为模型可以记住测试数据并在测试期间简单地将其regurgitate回来。在实践中查找并删除所有训练和测试数据重叠是很困难的。

参考文献：

[Language models are few-shot learners](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) GPT-3 作者 Brown 等人，将重叠示例定义为与预训练集中的任何其他示例共享至少 13 个连续单词的示例。如果一个示例少于 13 个单词，并且与另一个示例共享所有单词，则他们会认为该示例重叠。

[Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus](https://arxiv.org/abs/2104.08758) 中在网络爬取的 C4 语料库中搜索测试数据，但测量精确匹配，并对大写和标点符号进行标准化。他们发现文本生成和知识完成任务存在各种输入和标签污染；以及 GLUE 基准的仅输入污染。他们认为，测试数据可以通过两种方式最终出现在 Common Crawl（C4 的原始转储源）快照中：给定的测试集是从网络文本构建的，或者是在创建后上传的。

[Did ChatGPT cheat on your test?](https://hitz-zentroa.github.io/lm-contamination/blog/) 要求 ChatGPT 生成学术基准实例，发现它已经记住了多个实例，包括一些测试分割。

[Stop Uploading Test Data in Plain Text: Practical Strategies for Mitigating Data Contamination by Evaluation Benchmarks](https://arxiv.org/abs/2305.10160) 提出三种减轻污染的策略，包括encryption和training exclusion controls。

#### **Personally Identifiable Information (PII)**

在预训练语料库中发现了诸如电话号码和电子邮件地址之类的信息，导致提示过程中的隐私泄露。

#### Pre-Training Domain Mixtures

一些研究主张预训练语料库的多样性，许多流行的语料库都遵循这一点，将来自不同来源的数据集连接起来，然而，对于强大的下游性能需要多少不同来源的数据量仍然没有得到充分探索。

<figure><img src="../../.gitbook/assets/image (2).png" alt=""><figcaption><p>Table 1: <strong>Overview of Selected Pre-Training Datasets.</strong> Over the years, pre-training datasets have become more unfathomable: they grew rapidly in size and diversity, and not all datasets are publicly available (we do not include datasets that have very little or no information available about them). Unless stated otherwise, the natural language is in English. ∗ We report the number of tokens as provided by the respective paper based on their proposed tokenization scheme.预训练数据集变得更加深不可测：它们的大小和多样性迅速增长，并且并非所有数据集都是公开可用的.</p></figcaption></figure>

#### **Fine-Tuning Task Mixtures**

确定微调任务混合以便在许多不同的任务上微调预训练的模型，通常每个任务的示例相对较少。这种技术，我们称之为multitask-prompted fine-tuned(MTLM 多任务提示微调)，已经证明了在只需很少的额外训练计算的情况下显着的泛化改进。

如何很好地平衡任务数据集仍然不清楚。现在的一些做法是去模仿闭源model的数据配比，但是经过微调的开源模型和闭源模型之间仍然存在巨大的能力差距，这激励了未来研究更好的模仿（闭源模型）数据的工作。

### Challenge2：Tokenizer-Reliance 大语言模型的训练和运行通常依赖于特定的分词器，这可能对其性能和适应性产生影响。 --分词器带来了一些挑战，例如计算开销、语言依赖性、新词的处理、固定词汇量、信息丢失和低人类可解释性。

#### The number of tokens necessary to convey the same information varies significantly across languages 不同语言传递出相同的信息所需要的token数差距很大&#x20;

API 语言模型的定价政策（根据处理或生成的代币数量向用户收费）可能不公平。他们发现，许多受支持语言的用户在收到低于标准的结果的同时却被收取了过高的费用。

#### **Discrepancies between the data that a tokenizer and a model have been trained on can lead to glitch tokens  分词器和预训练语料库之间的不一致性可能导致错误 token**，进而导致模型行为异常

这可能会导致意外的模型行为，因为它们相应的embeddings基本上未经训练。每次修改预训练语料库时，分词器和预训练语料库之间的这种coupling（耦合）都会产生tokenizer新训练运行的负担。

#### 在多语言环境中运行良好的标记化方案，特别是对于非空格分隔的语言（例如中文或日语），仍然具有挑战性

现有的子词标记化方案主要是贪婪算法，试图根据所使用的标记数量尽可能有效地对语言进行编码。这些方法有利于包含大部分训练数据的subwords，因此也有利于在多种语言之间共享的subwords。这有利于具有shared scripts的语言，例如拉丁语和西里尔语，导致低资源语言的标记化效果不佳。

#### 此外，分词器会带来**计算负担、语言依赖性、处理新词、固定词汇表大小、信息丢失和人类可解释性**等多个挑战。

<figure><img src="../../.gitbook/assets/image (7).png" alt=""><figcaption><p><strong>Figure 2:</strong> Exemplary Drawbacks of relying on Tokenization. (1) The tokenizer training step involves non-trivial computations, e.g., multiple passes over the entire pre-training dataset, and introduces a dependency on it, which can become especially problematic in multilingual settings. (2) The embedding layer E and output layer W of LLMs involve the vocabulary size; e.g., making up ≈ 66% of the model’s parameter count in T5 models .</p></figcaption></figure>

如图 2 所示为依赖分词的典型缺点，其中，**分词器训练步骤涉及到复杂的计算**，如对整个预训练数据集进行多次扫描，并引入了对其的依赖，这在多语言环境下可能会变得特别棘手。此外，**语言模型的嵌入层 E 和输出层 W 涉及词汇表大小**，比如在 T5 模型中约占模型参数总数的 66% 左右。

针对这个挑战，子词级别的输入则提供了词汇大小和序列长度之间的良好平衡。此外，Byte-Pair Encoding (BPE)和 WordPiece 是常用的子词分词算法。字节级输入是子词分词的一种替代方法，可以与子词分词器结合使用或定义一个有限的词汇表来编码所有可能的序列。还有一些研究提出了基于字节级输入的分词方法，在性能方面与基于子词的模型相媲美。

### Challenge3：High Pre-Training Costs

大型语言模型的训练需要大量的计算资源和时间，这可能会对其广泛应用产生限制--不可持续

训练 LLM 的主要消耗是在预训练过程中，需要**数十万个计算小时、数百万元的成本，以及相当于数个普通美国家庭年度能源消耗量的能量**。而近期提出的缩放定律认为，**模型性能随着模型大小、数据集大小和训练中使用的计算量呈幂律关系，**将误差从 3% 减少到 2% 可能需要一个数量级的数据或计算。

Unsustainable Loss Power-Law：通过增加计算预算，性能会提高，但如果模型或数据集大小固定，性能会降低，这反映了收益递减的幂律。

<mark style="background-color:green;">**现有的两条解决路线**</mark>：

**Compute-Optimal Training Recipes**--**计算最优训练方法**

_Given a particular budget, how large should the pretraining corpus and model be to maximize training efficiency?_

学习经验性的“缩放定律“scaling laws”，以实现在给定计算预算下最大化训练效率

例如，OpenAI GPT4的报告称，他们能够根据一系列较小模型的性能准确预测全尺寸 GPT-4 模型的模型性能，使用的计算量最多比完整模型少 10,000 倍。

性能预测的一个剩余障碍是**inverse scaling**，由于缩放法则通常是在预训练的背景下构建的，从而与下游任务解耦，因此如何预测逆缩放属性仍然是一个悬而未决的问题。发现上游和下游设置中的缩放法则可能不同；除了模型大小之外，model shape对于下游微调也很重要。

**Pre-Training Objectives--预训练目标**

各种预训练目标（PTO）适合对LLM进行自我监督培训。 PTO 的准确选择会严重影响模型在预训练期间的数据效率，从而减少所需的迭代次数。

A PTO typically is a <mark style="color:green;">function</mark> of the <mark style="background-color:yellow;">(i) architecture,</mark> <mark style="background-color:yellow;">(ii) input/targets construction</mark> (e.g., t<mark style="background-color:yellow;">arget span length, low/high corruption, see Fig. 4,</mark> and (<mark style="background-color:yellow;">iii) masking strategy (Fig. 3).</mark> While (i) and (ii) can be disentangled and should not be conflated conceptually, in practice, there exist popular combinations that achieve good performances.

PTO通常是(i)架构，(ii)输入/目标结构(例如，目标跨度长度，低/高corruption (fig 4 )和(iii)掩蔽策略( fig 3)的函数。

<figure><img src="../../.gitbook/assets/Screenshot 2023-07-24 at 8.36.23 PM.png" alt="" width="563"><figcaption><p>Figure 3: Masking Strategies. Each row denotes to which inputs xi (columns) a particular output yi (row) can attend to (uni- or bi-directional). 屏蔽策略。每一行表示可以attend to 的输入(列)的一个特定的输出(行)(uni -或bi-directional)</p></figcaption></figure>

加入所有的token,如fig3所示(左),是最data-efficient策略,因为它使用上下文token之前和之后的预测。然而,由于这个原因,它不适合文本生成,因为它在预测中考虑了future context。我们通常用它来完成自然语言理解(NLU)任务。Next token prediction objective预测目标是最适合自然语言生成(NLG),但也data-efficient也是最低的,因为它只关注过去的上下文(fig3(中))。最近的研究目标是是找到一个middle-ground来提高效率的数据提供更强大和更多样化的训练信号,例如,Prefix LM,这中策略部分的关注past tokens,如fig3中所示(右)和下面讨论到的部分。

**Masked Language Modeling (MLM; or Cloze)**：通过用特殊的 \[MASK] 标记替换隐藏sequence中一定比例的标记。一般将 MLM 目标用于non-autoregressive，即non-generative、bidirectional context models。其中模型使用目标token之前和之后的token进行预测，有着对其上下文比 NTP 目标更全面的理解。此外，我们可以使用每个输入句子在一次传递中预测多个屏蔽标记，而 NTP 目标通常通过一次预测一个标记来学习。

[Bidirectional Language Models Are Also Few-shot Learners](https://arxiv.org/abs/2209.14500)中表明此类模型产生的表示更适合迁移学习；然而，它们在进行in-context learning时遇到困难。

一种提升方式:[METRO: Efficient Denoising Pretraining of Large Scale Autoencoding Language Models with Model Generated Signals](https://arxiv.org/abs/2204.06644) --(i) 使用 MLM 目标训练 ALM(auxiliary language model)，(ii) 给定一些带有masked positions的输入，预测tokens（使用 ALM），(iii) 训练主模型以纠正插入masked positions的这些tokens，即 1) 预测 ALM 是否替换了标记，如果是，2) 预测原始标记。train the auxiliary and main model **jointly**.

**Prefix Language Modeling**&#x20;

Generalizes language modeling by allowing prefix tokens with a <mark style="background-color:yellow;">bidirectional receptive field</mark> to be added to the input (without prefix, it is equivalent to standard LM). 对比fig3左和右

**Span Corruption** or span denoising

将 MLM 推广到对给定文本中的连续标记序列（称为spans）进行去噪。去噪目标通常用单个唯一的掩蔽标记替换采样的span，并训练模型来填充它。可以提升训练的速度，因为：与破坏 i.i.d. 中的单个token相比，span corruption平均会产生更短的序列。

**Mixture of Denoisers（MoD）**

通过混合多个去噪目标来注入目标多样性

**Fill In the Middle**

通过在文档中打乱token来增强下一个token预测目标，以便我们根据前缀和后缀填充中间（FIM）

**Meet in the Middle**

通过启用双向上下文来构建更密集、数据效率更高的监督信号，同时保持自回归，从而扩展 FIM 目标

**Parallelism Strategies**

**Miscellaneous**

<figure><img src="../../.gitbook/assets/image (4).png" alt=""><figcaption><p>Figure4：Self-Supervised Data Construction by Pre-Training Objectives. 通过预训练目标进行自我监督的数据构建，用灰色矩形表示屏蔽标记，这些标记成为目标，省略了特殊的标记。</p></figcaption></figure>

### Challenge:4：Fine-Tuning Overhead

对海量且多样化的文本数据集进行预训练LLM的一个潜在缺点是，生成的模型可能难以明确捕获特定于任务的数据集的分布属性。使用微调解决上述问题-->微调是指在特定于单个领域或任务的相对较小的数据集上调整预先训练的模型参数。 LLM fine-tuning对于使 LLM 适应下游任务非常有效 。

从技术上讲，可以通过在较小的数据集上进一步训练模型来实现微调。根据模型架构，可以通过以下方式完成的：(i) 使用标准语言建模目标直接微调预训练模型，(ii) 将单独的可学习层添加到预训练语言模型的输出表示中，这些层旨在在模型的输出表示和各个下游任务的输出格式（例如，用于文本分类或序列标记）之间创建兼容性。然而，<mark style="background-color:red;">拥有数十亿参数的LLM需要大量内存来存储（i）the model parameters，（ii）the model activations，以及（iii）gradients and corresponding statistics。由于设备内存（例如 GPU 或 TPU）有限，需要访问具有许多设备的大型集群来微调完整的 LLM。</mark>

#### Large Memory Requirements

<mark style="background-color:red;">微调整个 LLM 需要与预训练相同数量的内存</mark>，这对于许多从业者来说是不可行的。

此外，虽然完整的模型微调可以有效地使 LLM 在特定的下游任务上表现良好，但需要为各个任务存储和加载微调后的 LLM 的各个副本，这在计算上效率低下，（见fig5）

<figure><img src="../../.gitbook/assets/image (5).png" alt="" width="365"><figcaption><p>Figure 5: Fine-tuning an LLM for a specific downstream task. (a) illustrates vanilla fine-tuning, which requires updating the entire model, resulting in a new model for each task. In (b), PEFT instead learns a small subset of model parameters for each task with a fixed base LLM. The same base model can be re-used during inference for different tasks. <strong>将传统finetuning和peft做了对比。</strong></p></figcaption></figure>

#### **Parameter-efficient fine-tuning**

参数高效微调使LLM适应特定数据集/领域的另一种方法是通过参数高效微调（PEFT）。 PEFT 是指通过仅更新一小部分模型参数来适应 LLM 的一类方法。 [Adapters](https://arxiv.org/abs/1902.00751)是 PEFT 最早的工作之一。该方法将额外的可学习层合并到 Transformer 架构中，这些层在微调期间进行更新，同时保持网络的其余部分不变。 26 个文本分类任务（包括 GLUE 基准 ）的实验结果表明，通过 Adapter 训练的模型在仅更新 3% 的模型参数的情况下可以与full fine-tuning 竞争。同时，[BIt-Fit](https://aclanthology.org/2022.acl-short.1/)建议仅更新模型的偏差项进行微调，这些偏差项占模型参数的比例不到 1%。

目前Adapter合并到语言模型微调中的三个通用框架AdapterHub：[https://adapterhub.ml/](https://adapterhub.ml/)

* LLM-Adapters：[https://github.com/AGI-Edgerunners/LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters)
* HuggingFace  PEFT  ：[https://github.com/huggingface/peft](https://github.com/huggingface/peft)

针对较大模型引入的 PEFT 方法包括prefix-tuning和prompt-tuning，它们都是通过在输入前面添加一组可学习的token embeddings来进行操作。这些token embeddings（也称为soft prompt）是在微调阶段学习的，而模型参数的其余部分保持固定。最值得注意的是，这种软提示包含数千个而不是数百万个参数，并且存储效率更高。在微调令牌的同时仍然需要通过网络进行反向传播

仅具有API 访问权限的黑盒模型的替代方案：[Black-Box Prompt Learning for Pre-trained Language Models](https://arxiv.org/abs/2201.08531) & [Black-Box Tuning for Language-Model-as-a-Service](https://proceedings.mlr.press/v162/sun22e.html)

[Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638) 引入 $$(IA)^3$$ ，它使用可学习向量来缩放各个 Transformer 层中的激活。作者通过证明使用$$(IA)^3$$ 训练的模型优于各种数据集上的完整模型微调，同时仅更新 0.01% 的模型参数，证明了其有效性。

[Fine-Tuning Language Models with Just Forward Passes](https://arxiv.org/abs/2305.17333)提出了一种内存高效的零阶（MeZO）优化器，它只需要与推理期间相同的内存占用（而不是存储梯度或优化器状态）。此外，它还可以优化不可微分的目标，例如准确性或 F1 分数，传统的基于梯度的调整方法则无法做到这一点。

[LoRA: Low-Rank Adaptation of Large Language Models](./)提出低秩适应（LoRA），它将各个 Transformer 层权重矩阵的参数更新作为加法低秩分解。这种重新参数化避免了计算密集矩阵乘法的需要。QL0RA将 LoRA 扩展到量化的 LLM，大大减少了内存使用量，使他们能够在单个 48GB GPU 上微调 65B 模型，同一模型的常规训练需要超过 780 GB 的 GPU 内存。

**Compute Requirements**  针对特定任务微调 LLM 所需的内存复杂度有了显着提高，剩下的挑战是时间复杂度。使用 PEFT 方法对 LLM 进行微调，<mark style="background-color:red;">仍然需要完整的梯度计算</mark>。适应LLM所需的计算基础设施阻碍了小型设备上的个性化等潜在应用。



**Reference**

[Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)

[BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models](https://aclanthology.org/2022.acl-short.1.pdf)

[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353.pdf)

[The Power of Scale for Parameter-Efficient Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243.pdf)

[Black-Box Tuning for Language-Model-as-a-Service](https://proceedings.mlr.press/v162/sun22e.html)

[Black-Box Prompt Learning for Pre-trained Language Models](https://arxiv.org/abs/2201.08531)

[Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638)

[Fine-Tuning Language Models with Just Forward Passes](https://arxiv.org/abs/2305.17333)

[LoRA: Low-Rank Adaptation of Large Language Models](./)



### Challenge5：High Inference Latency 推理延迟

目前根据研究，LLM 表现出高推理延迟的两个原因：

(1) **low parallelizability** 并行性低，因为推理过程一次只进行一个token；

(2) **large memory footprints** 内存占用较大--due to the model size and the transient states needed during decoding (e.g., attention key and value tensors).

此外， Transformers 中注意力机制的二次缩放也是值得讨论的，见challlenge6。

#### **Techniques used to address these challenges eg.** 减少内存占用（大小和/或带宽&加速特定的计算操作

<mark style="background-color:blue;">**Efficient Attention   --加速attention机制的计算**</mark>

<mark style="color:green;">Two lines of work： (i) lower-level hardware-aware modifications 较低级别的硬件感知modifications (ii) higher-level sub-quadratic approximations of the attention mechanism 注意力机制更高层次的次二次方近似</mark>

**Lower-level hardware-aware modifications**

1. [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) 提出 <mark style="color:blue;">multi-query attention</mark> 旨在通过为键和值张量仅保留一个注意力头，在使用 Transformer 解码器层顺序生成token序列时减少内存带宽瓶颈。
2.  [FlashAttention](https://arxiv.org/abs/2205.14135) 的多头自注意力替代计算方法来减少内存带宽，以最大限度地减少 I/O 操作的数量，从而加快现代 GPU 上的计算速度。作为一种优化的注意力， FlashAttention 利用算子融合来减少内存带宽瓶颈。&#x20;

    [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) 提出`FlashAttention`

    [Faster Causal Attention Over Large Sequences Through Sparse Flash Attention](../lian-dan-gong-ju-xiang/paramters-and-definations.md) 构建在 FlashAttention 之上，并结合注意力稀疏模式--key/query dropping and hashing-based attention。

    [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) 实施不同的sharding技术，以有效地跨设备传播前馈和注意力计算，同时优化设备间通信成本，**使用muli-query attention实现高达 43,000 个token的上下文长度。**

**Higher-level sub-quadratic approximations of the attention mechanism**

提高注意力机制的计算或内存复杂度的一个共同主题是稀疏注意力矩阵或引入（线性）近似。

然而，一些有效的注意力近似的可扩展性受到了质疑。比如例如，有研究发现发现Performer attention approximation 近似的表现严重低于普通的自注意力机制，特别是当扩展到大型模型时。

<mark style="background-color:blue;">**Quantization**</mark>

量化是一种post-training技术，可通过降低权重和激活的计算精度来减少内存占用和/或增加模型的吞吐量。[ nuQmm](https://arxiv.org/abs/2206.09557) 和 [ZeroQuant ](https://arxiv.org/abs/2206.01861)使用非均匀量化方法来量化权重并应用自定义 CUDA 内核以获得计算优势。 [LLM.int8() ](https://arxiv.org/abs/2208.07339) 是一种无降级量化方案，通过利用 Int8 量化来有效推断数十亿参数 LLM，并针对某些outlier特征回退到更高的精度，而无需重新训练。

类似地，[GLM-130B](https://arxiv.org/abs/2210.02414) 使用无降级8位量化方案，以8位存储权重并以16位精度执行矩阵乘法。[GPTQ](https://arxiv.org/abs/2210.17323)是一种高效的一次性量化技术，将 LLM 权重压缩至每个权重 3 到 4 位，从而使 175B 参数模型能够在单 个 GPU 上运行。[SpQR](https://arxiv.org/abs/2306.03078) 通过结合异常值权重的更高精度表示和分组量化来进一步改进这一点。

<mark style="background-color:blue;">**Pruning**</mark>

Pruning是用于量化的complementary post-training 技术，要删除给定模型的部分权重（不会降低其性能）。一个重要的区别是修剪是遵循结构化模式还是非结构化模式。结构化稀疏模型用明显更小但仍然密集的组件的集合来替代模型的密集部分。非结构化稀疏模型包含的权重为零，这不会影响网络的行为，因此理论上可以接受。然而，在实践中，在当前硬件上将理论转化为实际计算节省更具挑战性。



**Reference**

[Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)

[Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)



### 相关解读：

* [https://mp.weixin.qq.com/s/JsCoUcuCg4ylKkPMvNouEw](https://mp.weixin.qq.com/s/JsCoUcuCg4ylKkPMvNouEw)



