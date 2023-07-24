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

<figure><img src="../../.gitbook/assets/image.png" alt=""><figcaption><p>Table 1: <strong>Overview of Selected Pre-Training Datasets.</strong> Over the years, pre-training datasets have become more unfathomable: they grew rapidly in size and diversity, and not all datasets are publicly available (we do not include datasets that have very little or no information available about them). Unless stated otherwise, the natural language is in English. ∗ We report the number of tokens as provided by the respective paper based on their proposed tokenization scheme.预训练数据集变得更加深不可测：它们的大小和多样性迅速增长，并且并非所有数据集都是公开可用的.</p></figcaption></figure>

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

#### High Pre-Training Costs

大型语言模型的训练需要大量的计算资源和时间，这可能会对其广泛应用产生限制--不可持续

训练 LLM 的主要消耗是在预训练过程中，需要**数十万个计算小时、数百万元的成本，以及相当于数个普通美国家庭年度能源消耗量的能量**。而近期提出的缩放定律认为，**模型性能随着模型大小、数据集大小和训练中使用的计算量呈幂律关系，**将误差从 3% 减少到 2% 可能需要一个数量级的数据或计算。

Unsustainable Loss Power-Law：通过增加计算预算，性能会提高，但如果模型或数据集大小固定，性能会降低，这反映了收益递减的幂律。

<mark style="background-color:green;">现有的两种解决路线</mark>：

**Compute-Optimal Training Recipes**--**计算最优训练方法**

_Given a particular budget, how large should the pretraining corpus and model be to maximize training efficiency?_

学习经验性的“缩放定律“scaling laws”，以实现在给定计算预算下最大化训练效率

例如，OpenAI GPT4的报告称，他们能够根据一系列较小模型的性能准确预测全尺寸 GPT-4 模型的模型性能，使用的计算量最多比完整模型少 10,000 倍。

性能预测的一个剩余障碍是**inverse scaling**，由于缩放法则通常是在预训练的背景下构建的，从而与下游任务解耦，因此如何预测逆缩放属性仍然是一个悬而未决的问题。发现上游和下游设置中的缩放法则可能不同；除了模型大小之外，model shape对于下游微调也很重要。

**Pre-Training Objectives--预训练目标**

各种预训练目标（PTO）适合对LLM进行自我监督培训。 PTO 的准确选择会严重影响模型在预训练期间的数据效率，从而减少所需的迭代次数。

<figure><img src="../../.gitbook/assets/image (4).png" alt=""><figcaption><p>Figure4：Self-Supervised Data Construction by Pre-Training Objectives. 通过预训练目标进行自我监督的数据构建，用灰色矩形表示屏蔽标记，这些标记成为目标，省略了特殊的标记。</p></figcaption></figure>



相关解读：

* [https://mp.weixin.qq.com/s/JsCoUcuCg4ylKkPMvNouEw](https://mp.weixin.qq.com/s/JsCoUcuCg4ylKkPMvNouEw)

