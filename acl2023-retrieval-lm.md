# [ACL 2023 Tutorial:Retrieval-based Language Models and Applications](https://acl2023-retrieval-lm.github.io/)

Akari Asai, Sewon Min, Zexuan Zhong, Danqi Chen

这里主要中文总结本教程中的一些重点内容

讲者说明：
本教程是最前沿的，与参数化llm相比，我们还远远不能理解如何最好地开发基于检索的lm，这个教程主要分享：
* 现有研究的分类和关键见解
* 我们对当前挑战和开放问题的看法

## 1. Introduction

**1. 什么是Retrieval-based language models (LMs)？**

Retrieval-based LMs = Retrieval + LMs

![LM  retrieves from an external datastore (at least during inference time)](./figure/image16.png)
语言模型从外部数据存储中进行检索（至少在推理期间）这样的模型也被称为半参数模型和非参数模型（semiparametric 
and non-parametric models）

**2. The age of large language models (LLMs)：主要介绍目前大语言模型的一些特点**
   * Transformers-based, fully parametric 
   * Trained on next-token prediction tasks (+ RLHF;)
   * Model size ↑, data size↑


**3. Retrieval for knowledge-intensive NLP tasks** 对知识密集型任务的检索
   
   **Representative tasks**: open-domain QA, fact checking, entity linking...

   LM推动了大量关于密集检索的更好算法的研究，例如，DPR，ColBERT,ANCE,Contriever，..


**4. Why** retrieval-based LMs?
   
  	* LLMs can’t memorize all (long-tail) knowledge in their parameters 大模型的参数对知识的记忆有限
	* LLMs’ knowledge is easily outdated and hard to update 大模型的知识容易过时，难以更新--**现有的知识编辑方法仍然是不可扩展的**（研究方向！）而数据存储可以很容易地更新和扩展——甚至不需要重新训练模型
	* LLMs’ output is challenging to interpret and verify 大模型的输出难以验证和解释--从检索结果中更新知识来源可以获得更好的解释性和控制性（Generating text with citations，like newbing）
	* LLMs are shown to easily leak private training data 大模型容易泄漏私有训练数据 ，所以可以通过将私人数据存储在数据存储器中，从而对其进行个性化处理（而不是直接参与模型参数训练？）
	* LLMs are *large* and expensive to train and run 大模型训练和运行成本高，而数据存储器可以在推理期间进行检索，因此可以**减少模型的大小和成本** --Long-term goal: can we possibly reduce the training and inference costs, and scale down the size of LLMs?


## 2. Definition & Preliminaries

1. A Retrieval-based LM: Definition - A language model (LM) that usesan external datastore at test time  在测试期间使用外部数据存储的语言模型
2. A language model (LM): Categories
	 ![Alt text](/figure/image17.png)
	 这里有一个问题是**为什么Decoder-only模型几乎成为了现在LLM的主流架构**？
	 参考博客：https://kexue.fm/archives/9529  https://www.zhihu.com/question/588325646
	 
	 主要观点:任何NLP任务都可以分解为“输入”跟“输出”两部分，我们可以把处理“输入”的模型叫做Encoder，生成“输出”的模型叫做Decoder，那么所有任务都可以从“Encoder-Decoder”的视角来理解，而不同模型之间的差距在于Encoder、Decoder的注意力模式以及是否共享参数,比如：
	| Model |  Encoder 注意力 |  Dncoder 注意力 | 是否共享参数 |
	|-------|--------------|--------------|--------|
	| GPT   | 单向           | 单向           | 是      |
	| UniLM | 双向           | 单向           | 是      |
	| T5    | 双向           | 单向           | 否      |

	这里的GPT就是Decoder-only的代表作；UniLM则是跟GPT相似的Decoder架构，但它是混合的注意力模式；T5则是Encoder-Decoder架构的代表作，主要是Google比较感兴趣。

	1. GPT和UniLM的对比实验，结果说明：“**输入部分的注意力改为双向不会带来收益，Encoder-Decoder架构的优势很可能只是源于参数翻倍。**”
   
	2. **低秩问题**：众所周知，Attention矩阵一般是由一个低秩分解的矩阵加softmax而来，具体来说是一个n×d的矩阵与d×n的矩阵相乘后再加softmax（n≫d），这种形式的Attention的矩阵因为低秩问题而带来表达能力的下降，具体分析可以参考《Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth》。而Decoder-only架构的Attention矩阵是一个下三角阵，注意三角阵的行列式等于它对角线元素之积，由于softmax的存在，对角线必然都是正数，所以它的行列式必然是正数，即Decoder-only架构的Attention矩阵一定是满秩的，满秩意味着理论上有更强的表达能力，也就是说，Decoder-only架构的Attention矩阵在理论上具有更强的表达能力，改为双向注意力反而会变得不足。（可以参考线性attention相关内容：《Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth》&《Transformer升级之路》）
	3. 《Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth》
   
		论文最主要的一个结论是decoder-only模型在没有任何tuning数据的情况下、zero-shot表现最好，而encoder-decoder则需要在一定量的标注数据上做multitask finetuning才能激发最佳性能。而目前的Large LM的训练范式还是在大规模语料上做自监督学习，很显然，Zero-Shot性能更好的decoder-only架构才能更好地利用这些无标注数据。此外，Instruct GPT在自监督学习外还引入了RLHF作辅助学习。RLHF本身也不需要人工提供任务特定的标注数据，仅需要在LLM生成的结果上作排序。虽然目前没有太多有关RLHF + encoder-decoder的相关实验，直觉上RLHF带来的提升可能还是不如multitask finetuning，毕竟前者本质只是ranking、引入监督信号没有后者强。

		即，encoder-decoder在multitask finetuning上的优势在大参数量时被LLM的推理能力给拉平了。

3. A language model (LM): Prompting
      	 
	通过不同的prompt使LM完成不通的任务

4. A language model (LM): Often evaluated with
	给出评价指标：1. Perplexity 2. Downstream accuracy (Zero-shot or few-shot in-context learning,or fine-tuning) 会在第五节详细介绍

	一个问题：**为啥要用perplexity（困惑度）来作为本课程的主要指标**
  
	“在比较参数化的语言模型时，困惑度（PPL）经常被用到。但困惑度的改善能否转化为下游应用仍然是一个研究问题。

	现已有研究表明，**困惑度与下游任务（尤其是生成任务）有很好的相关性，并且困惑度通常可提供非常稳定的结果，它可以在大规模评估数据上进行评估**（相对于下游任务来说，评估数据是没有标签的，而下游任务可能会受到提示的敏感性和缺乏大规模标记数据的影响，从而导致结果不稳定）。”
5. Inference: Datastore
	![Alt text](/figure/image18.png)

6. Inference: Index 
   
	目标：在数据存储中找到与查询最相似的一小部分元素

    **sim**：a similarity score between two pieces of text 下面是similarity score的一些例子
	 ![Alt text](/figure/image19.png)
	**index**：给定query，通过fast nearest neighbor search（这也是一个研究方向-如何更加快快速和准确），输出sim最大的k个元素

	相关software: FAISS, Distributed FAISS, SCaNN, etc…

	参考：https://github.com/facebookresearch/faiss/wiki

## 3. Retrieval-based LM: Architecture

