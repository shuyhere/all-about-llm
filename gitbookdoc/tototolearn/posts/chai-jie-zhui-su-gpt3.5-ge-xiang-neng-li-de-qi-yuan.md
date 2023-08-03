---
description: yao fu
---

# 🔆 拆解追溯 GPT-3.5 各项能力的起源

原文链接：[https://yaofu.notion.site/GPT-3-5-360081d91ec245f29029d37b54573756](https://yaofu.notion.site/GPT-3-5-360081d91ec245f29029d37b54573756)

上下文学习：[https://thegradient.pub/in-context-learning-in-context/](https://thegradient.pub/in-context-learning-in-context/)

## 原文（+一些笔记）

[**符尧**](https://franxyao.github.io), [yao.fu@ed.ac.uk](mailto:yao.fu@ed.ac.uk)

爱丁堡大学博士生，硕士毕业于哥伦比亚大学，本科毕业于北京大学

与 [**彭昊**](https://haopeng-nlp.github.io)，[\*\*Tushar Khot](https://allenai.org/team/tushark)\*\*

在艾伦人工智能研究院 (Allen Institute for AI) 共同完成英文原稿

与 剑桥大学[**郭志江**](https://cartus.github.io) 共同翻译为中文

感谢 上海交通大学[**何俊贤**](https://jxhe.github.io)，加州大学洛杉矶分校[**鲁盼**](https://lupantech.github.io)，达特茅斯学院[**刘睿博**](https://www.cs.dartmouth.edu/\~rbliu/) 对初稿的讨论与建议。

感谢 [**Raj Ammanabrolu**](http://prithvirajva.com) (Allen Institute for AI), [**Peter Liu**](https://peterjliu.com) (Google Brain), [**Brendan Dolan-Gavitt**](https://www.notion.so/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1?pvs=21) (New York University), [**Denny Zhou**](https://dennyzhou.github.io) (Google Brain) 对终稿的讨论和建议，他们的建议极大程度上增加了本文的完整度。

英文版完稿于 2022 年 12 月 11 日，中文版完稿于 2022 年 12 月 18 日。

其他版本: \[pdf] \[Arxiv] \[[英文原版](https://www.notion.so/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1?pvs=21)] \[学术引用]

在[推特](https://twitter.com/Francis\_YAO\_/status/1602213927102066688?s=20\&t=9wkRcr0wva\_RCaKpsRjFfw)上与作者互动

初次翻译，哪里没写好，不地道的地方，还请邮件帮忙指出

转发请在文章的开头标明出处，而不是在结尾列一行小字



最近，OpenAI的预训练模型ChatGPT给人工智能领域的研究人员留下了深刻的印象和启发。毫无疑问，它又强又聪明，且跟它说话很好玩，还会写代码。它在多个方面的能力远远超过了自然语言处理研究者们的预期。于是我们自然就有一个问题：ChatGPT 是怎么变得这么强的？它的各种强大的能力到底从何而来？在这篇文章中，我们试图剖析 ChatGPT 的突现能力（Emergent Ability），追溯这些能力的来源，希望能够给出一个全面的技术路线图，来说明 GPT-3.5 模型系列以及相关的大型语言模型是如何一步步进化成目前的强大形态。

我们希望这篇文章能够促进大型语言模型的透明度，成为开源社区共同努力复现 GPT-3.5 的路线图。

Recently, the field has been greatly impressed and inspired by OpenAI’s ChatGPT. It is undoubtedly clever, capable, and very fun to talk to. Its multi-faceted abilities are significantly beyond many NLP researchers’ and practitioners’ expectations based on the impression of (not-that-strong) original GPT-3. The natural question is how ChatGPT gets there, and where these fantastic abilities come from. In this post, we try to dissect the emergent abilities and trace them to their sources, hoping to give a comprehensive roadmap about how the GPT-3.5 model family, along with related large language models, evolved to their current forms.

We hope this post can promote the transparency of large language models and serve as the roadmap for the community’s ongoing efforts of reproducing GPT-3.5.

### &#x20;一、2020 版初代 GPT-3 与大规模预训练

初代GPT-3展示了三个重要能力：

* **语言生成**：遵循提示词（prompt），然后生成补全提示词的句子 (completion)。这也是今天人类与语言模型最普遍的交互方式。
* **上下文学习 (in-context learning)**: 遵循给定任务的几个示例，然后为新的测试用例生成解决方案。很重要的一点是，GPT-3虽然是个语言模型，但它的论文几乎没有谈到“语言建模” (language modeling) —— 作者将他们全部的写作精力都投入到了对上下文学习的愿景上，这才是 GPT-3的真正重点。
* **世界知识 (world knowledge)**：包括事实性知识 (factual knowledge) 和常识 (commonsense)。

那么这些能力从何而来呢？

基本上，以上三种能力都来自于大规模预训练：在有3000亿单词的语料上预训练拥有1750亿参数的模型（ 训练语料的60%来自于 2016 - 2019 的 C4 + 22% 来自于 WebText2 + 16% 来自于Books + 3%来自于Wikipedia）。其中：

* **语言生成**的能力来自于语言建模的**训练目标** (language modeling)。
* **世界知识**来自 3000 亿单词的**训练语料库**（不然还能是哪儿呢）。
* **模型的 1750 亿参数**是为了**存储知识**，Liang et al. (2022) 的文章进一步证明了这一点。 他们的结论是，[知识密集型任务的性能与模型大小息息相关](https://crfm.stanford.edu/helm/v0.2.2/?group=knowledge)。
* 上下文学习的能力来源及为什么上下文学习可以泛化**仍然难以溯源** 直觉上，<mark style="background-color:yellow;">这种能力可能来自于同一个任务的数据点在训练时按顺序排列在同一个 batch 中</mark>。然而，[<mark style="background-color:red;">很少有人研究为什么语言模型预训练会促使上下文学习，以及为什么上下文学习的行为与微调 (fine-tuning) 如此不同。</mark>](#user-content-fn-1)[^1]

There are three important abilities that the initial GPT-3 exhibit:

* **Language generation**: to follow a prompt and then generate a completion of the given prompt. Today, this might be the most ubiquitous way of human-LM interaction.
* **In-context learning**: to follow a few examples of a given task and then generate the solution for a new test case. It is interesting to note that, although being a language model, the original GPT-3 paper barely talks about “language modeling” — the authors devoted their writing efforts to their visions of in-context learning, which is the real focus of GPT-3.
* **World knowledge**: including factual knowledge and commonsense.

Where do these abilities come from?

Generally, the above three abilities should come from large-scale pretraining — to pretrain the 175B parameters model on 300B tokens (60% 2016 - 2019 C4 + 22% WebText2 + 16% Books + 3% Wikipedia). Where:

* The **language generation** ability comes from the language modeling **training objective**.
* The **world knowledge** comes from the 300B token **training corpora** (or where else it could be).
* The **175B model size** is for **storing knowledge**, which is further evidenced by Liang et al. (2022), who conclude that the performance on tasks requiring knowledge correlates with model size.
* The source of the in-context learning ability, as well as its generalization behavior, is still elusive. Intuitively, this ability may come from the fact that data points of the same task are ordered sequentially in the same batch during pretraining. Yet there is little study on why language model pretraining induces in-context learning, and why in-context learning behaves so differently than fine-tuning.

令人好奇的是，初代**的GPT-3有多强。** 其实比较难确定初代 GPT-3（在 OpenAI API 中被称为`davinci`）到底是“强”还是“弱”。一方面，它合理地回应了某些特定的查询，并在许多数据集中达到了还不错的性能；另一方面，它在许多任务上的**表现还不如 T5 这样的小模型**（参见其原始论文）。在今天（2022 年 12 月）ChatGPT 的标准下，很难说初代的 GPT-3 是“智能的”。Meta 开源的 OPT 模型试图复现初代 GPT-3，但它的能力与当今的标准也形成了尖锐的对比。许多测试过 OPT 的人也认为与现在的`text-davinci-002`相比，该模型确实 “不咋地”。尽管如此，OPT 可能是初代 GPT-3 的一个足够好的开源的近似模型了（根据 OPT 论文和斯坦福大学的 HELM 评估）。

虽然初代的 GPT-3 可能表面上看起来很弱，但后来的实验证明，初代 GPT-3 有着非常强的潜力。这些潜力后来被代码训练、指令微调 (instruction tuning) 和基于人类反馈的强化学习 (reinforcement learning with human feedback, RLHF) 解锁，最终体展示出极为强大的突现能力。

A curious question is **how strong the initial GPT-3 is.**

It is rather challenging to determine whether the initial GPT-3 (`davinci` in OpenAI API) is “strong” or “weak.” On the one hand, it responds to certain queries reasonably and achieves OK-ish performance on many benchmarks; on the other, **it underperforms small models** like T5 on many tasks (see its original paper). It is also very hard to say the initial GPT-3 is “smart” in today's (= Dec 2022) ChatGPT standard. The sharp comparison of initial GPT-3’s ability v.s. today’s standard is replayed by Meta’s OPT model, which is viewed as “just bad” by many who have tested OPT (compared to `text-davinci-002`). Nevertheless, OPT might be a good enough open-source approximation to the initial GPT-3 (according to the OPT paper and Stanford’s HELM evaluation).

Although the initial GPT-3 might be superficially weak, it turns out later that these abilities serve as very important foundations of all the emergent abilities unlocked later by training on code, instruction tuning, and reinforcement learning with human feedback (RLHF).

### 二、从 2020 版 GPT-3 到 2022 版 ChatGPT

从最初的 GPT-3 开始，为了展示 OpenAI 是如何发展到ChatGPT的，我们看一下 GPT-3.5 的进化树：

<figure><img src="../../.gitbook/assets/image (8).png" alt=""><figcaption></figcaption></figure>

在 **2020 年 7 月**，OpenAI 发布了模型索引为的 `davinci` 的初代 [GPT-3 论文](https://arxiv.org/abs/2005.14165)，从此它就开始不断进化。在 **2021 年 7 月**，[Codex 的论文](https://arxiv.org/abs/2107.03374)发布，其中初始的 Codex 是根据（可能是内部的）120 亿参数的 GPT-3 变体进行微调的。后来这个 120 亿参数的模型演变成 OpenAI API 中的`code-cushman-001`。在 **2022 年 3 月**，OpenAI 发布了[指令微调 (instruction tuning) ](https://arxiv.org/abs/2203.02155)的论文，其监督微调 (supervised instruction tuning) 的部分对应了`davinci-instruct-beta`和`text-davinci-001`。在 **2022 年 4 月至 7 月的**，OpenAI 开始对`code-davinci-002`模型进行 Beta 测试，也称其为 Codex。然后text`-davinci-002`、`text-davinci-003`和`ChatGPT` 都是从`code-davinci-002`进行指令微调得到的。详细信息请参阅 OpenAI的模型索引文档。

尽管 Codex 听着像是一个只管代码的模型，但`code-davinci-002`可能是最强大的针对**自然语言**的GPT-3.5 变体（优于 `text-davinci-002`和 `-003`）。`code-davinci-002`很可能在文本和代码上都经过训练，然后根据指令进行调整（将在下面解释）。然后**2022 年 5-6 月**发布的`text-davinci-002`是一个基于`code-davinci-002`的有监督指令微调 (supervised instruction tuned) 模型。<mark style="background-color:red;">在</mark><mark style="background-color:red;">`text-davinci-002`</mark><mark style="background-color:red;">上面进行</mark><mark style="background-color:red;">**指令微调**</mark><mark style="background-color:red;">很可能</mark><mark style="background-color:red;">**降低**</mark><mark style="background-color:red;">了模型的</mark><mark style="background-color:red;">**上下文学习**</mark><mark style="background-color:red;">能力，</mark><mark style="background-color:red;">**但是**</mark><mark style="background-color:red;">增强了</mark><mark style="background-color:red;">**模型的**</mark><mark style="background-color:red;">零样本能力（将在下面解释）</mark>。然后是`text-davinci-003`和 `ChatGPT`，它们都在 **2022 年 11 月**发布，是使用的基于人类反馈的强化学习的版本指令微调 (instruction tuning with reinforcement learning from human feedback) 模型的两种不同变体。`text-davinci-003` 恢复了（但仍然比`code-davinci-002`差）一些在`text-davinci-002` 中丢失的部分**上下文学习能**力（大概是因为它在微调的时候混入了语言建模） 并进一步改进了零样本能力（得益于RLHF）。另一方面，[<mark style="background-color:red;">ChatGPT 似乎</mark><mark style="background-color:red;">**牺牲了几乎所有的上下文学习的能力**</mark><mark style="background-color:red;">来</mark><mark style="background-color:red;">**换取**</mark><mark style="background-color:red;">建模对话历史的能力。</mark>](#user-content-fn-2)[^2]

总的来说，在 2020 - 2021 年期间，在`code-davinci-002`之前，OpenAI 已经投入了大量的精力通过代码训练和指令微调来增强GPT-3。当他们完成`code-davinci-002`时，所有的能力都已经存在了。很可能后续的指令微调，无论是通过有监督的版本还是强化学习的版本，都会做以下事情（稍后会详细说明）：

* 指令微调**不会为模型注入新的能力** —— <mark style="background-color:red;">所有的能力都已经存在了。指令微调的作用是</mark><mark style="background-color:red;">**解锁 / 激发这些能力**</mark><mark style="background-color:red;">。这主要是因为指令微调的数据量比预训练数据量少几个数量级（基础的能力是通过预训练注入的）。</mark>
* 指令微调**将 GPT-3.5 的分化到不同的技能树**。有些更擅长上下文学习，如`text-davinci-003`，有些更擅长对话，如`ChatGPT`。
* 指令微调**通过牺牲性能换取与人类的对齐（alignment）**。 OpenAI 的作者在他们的指令微调论文中称其为 “对齐税” (alignment tax)。许多论文都报道了`code-davinci-002`在基准测试中实现了最佳性能（但模型不一定符合人类期望）。 在`code-davinci-002`上进行指令微调后，模型可以生成更加符合人类期待的反馈（或者说模型与人类对齐），例如：零样本问答、生成安全和公正的对话回复、拒绝超出模型它知识范围的问题。

In **Jul 2020**. OpenAI released the initial GPT-3 paper with the `davinci` model index, and it started to evolve. In **Jul 2021**, the Codex paper was released, where the initial Codex is fine-tuned from a (presumably internal) 12B GPT-3 variant. Later this 12B model evolved to be the `code-cushman-001` in OpenAI API. In **Mar 2022**, OpenAI released the instruction tuning paper, and its supervised tuning part corresponds to the `davinci-instruct-beta` and `text-davinci-001`. At some point in **Apr-Jul 2022**, OpenAI started to beta test the `code-davinci-002` model, also calling it Codex. Then `text-davinci-002`, `text-davinci-003`, and `ChatGPT` are all instruction-tuned from `code-davinci-002`. See OpenAI’s Model Index document for more details.

Although called Codex, **code**-davinci-002 is probably **the most capable** GPT-3.5 variant for **natural language** (better than text-davinci-002 and 003). It is very likely code-davinci-002 is trained on both text and code, then tuned on instructions (will explain below). Then text-davinci-002, released in **May-Jun 2022**, is a supervised instruction-tuned model based on code-davinci-002. It is very likely that the **instruction tuning** on text-davinci-002 **decreased** the model’s **in-context learning** ability but **increased** the model’s **zero-shot** ability (will explain below). Then text-davinci-003 and ChatGPT, both released in **Nov 2022**, are two different variants of instruction-tuned models using Reinforcement Learning with Human Feedback. text-davinci-003 **recovered** (but still worse than code-davinci-002) some **in-context learning** ability that is lost in text-davinci-002 (presumably because it tunes the model with LM mix-in) **and further improved zero-shot** ability (thanks to RLHF). On the other hand, ChatGPT seems to have **sacrificed nearly all** of its **in-context learning** ability to **trade for** the ability to model **dialog** context.

In summary, during 2020-2021, before code-davinci-002, substantial efforts have been devoted to enhancing GPT-3 with code training and instruction tuning. When they have reached code-davinci-002, all the abilities are there. It is likely that the following-up instruction-tuning, either supervised or RLHF, does the following things (will detail later):

* Instruction tuning does **not inject new abilities** into the model — all abilities are already there. Instead, instruction tuning **unlocks/ elicit these abilities**. This is mostly because the instruction tuning data is orders or magnitudes less than the pretraining data.
* Instruction tuning **adjusts skillsets** of GPT-3.5 **towards different branches**. Some are better at in-context learning like text-davinci-003, some are better at dialog like ChatGPT.
* Instruction tuning **trade performance for alignment** with humans. The OpenAI authors call it “alignment tax” in their instruction tuning paper. Also, many papers have reported code-davinci-002 achieves the best performance on benchmarks. Instruction tuning on code-davinci-002 gives the subsequent models alignments like zero-shot question answering, generating safe and impartial dialog responses, and rejecting questions beyond its knowledge scope.

### 三、Code-Davinci-002和 Text-Davinci-002，在代码上训练，在指令上微调

在`code-davinci-002`和`text-davinci-002`之前，有两个中间模型，分别是 davinci-instruct-beta 和 text-davinci-001。两者在很多方面都比上述的两个-002模型差（例如，text-davinci-001 链式思维推理能力不强）。所以我们在本节中重点介绍 -002 型号。

Before code-davinci-002 and text-davinci-002, there are two intermediate models, namely davinci-instruct-beta and text-davinci-001. Both are worse than the two -002 models in many aspects (e.g., text-davinci-001 cannot do chain-of-thought reasoning). So we focus on the -002 models in this section.

#### **3.1 复杂推理能力的来源和泛化到新任务的能力**

我们关注`code-davinci-002`和`text-davinci-002`，这两兄弟是第一版的 GPT3.5 模型，一个用于代码，另一个用于文本。它们表现出了四种与初代 GPT-3 不同的重要能力：

* **响应人类指令**：以前，GPT-3 的输出主要是训练集中常见的句子。现在的模型会针对指令 / 提示词生成更合理的答案（而不是相关但无用的句子）。
* **泛化到没有见过的任务**：当用于调整模型的指令数量超过一定的规模时，模型就可以自动在从没见过的新指令上也能生成有效的回答。 **这种能力对于上线部署至关重要**，因为用户总会提新的问题，模型得答得出来才行。
* **代码生成和代码理解**：这个能力很显然，因为模型用代码训练过。
* **利用思维链 (chain-of-thought) 进行复杂推理**：初代 GPT3 的模型思维链推理的能力很弱甚至没有。 **code-davinci-002 和 text-davinci-002 是两个拥有足够强的思维链推理能力的模型。**
  * 思维链推理之所以重要，是因为思维链可能是解锁突现能力和超越缩放法则 (scaling laws) 的关键。请参阅[上一篇博文](https://yaofu.notion.site/A-Closer-Look-at-Large-Language-Models-Emergent-Abilities-493876b55df5479d80686f68a1abd72f)。

Now let’s look at code-davinci-002 and text-davinci-002, the two first GPT3.5 models, one for code and the other for text. There are three important abilities they exhibit that differentiate them from the initial GPT-3

* **Responding to human instruction**: previously, the outputs of GPT-3 were mostly high-frequency prompt-completion patterns within the training set. Now the model generates reasonable answers to the prompt, rather than related but useless sentences.
* **Generalization to unseen tasks**: when the number of instructions used for tuning the model is beyond a certain scale, the model can automatically generate completions for new instructions that are not in the training set. **This ability is crucial for deployment**, as users with always come up with new prompts.
* **Code generation and code understanding**: obviously, because the model is trained on code.
* **Complex reasoning** **with chain-of-thought**: previously, the model could not do tasks requiring multi-step reasoning with chain-of-thought. **codex-davinci-002 and text-davinci-002 are the two initial models exhibiting chain-of-thought reasoning ability**.
  * The reason that chain-of-thought is important is because that CoT is likely to be the key to unlock the emergent abilities and transcend scaling laws. See the previous blog post.

这些能力从何而来？

与之前的模型相比，两个主要区别是**指令微调**和**代码训练**。具体来说

* 能够**响应人类指令**的能力是**指令微调**的直接产物。
* <mark style="background-color:red;">**对没有见过的指令做出反馈**</mark><mark style="background-color:red;">的泛化能力是在指令数量超过一定程度之后</mark><mark style="background-color:red;">**自动出现的**</mark><mark style="background-color:red;">，T0、Flan 和 FlanPaLM 论文进一步证明了这一点</mark>
* <mark style="background-color:red;">使用</mark><mark style="background-color:red;">**思维链**</mark><mark style="background-color:red;">进行</mark><mark style="background-color:red;">**复杂推理**</mark><mark style="background-color:red;">的能力很可能是</mark><mark style="background-color:red;">**代码训练**</mark><mark style="background-color:red;">的</mark><mark style="background-color:red;">**一个神奇的副产物**</mark><mark style="background-color:red;">。对此，我们有以下的事实作为一些支持：</mark>
  * 最初的 GPT-3 没有接受过代码训练，它不能做**思维链**。
  * text-davinci-001 模型，虽然经过了指令微调，但第一版思维链论文报告说，它的它思维链推理的能力非常弱 —— **所以指令微调可能不是思维链存在的原因，代码训练才是模型能做思维链推理的最可能原因。**
  * PaLM 有 5% 的代码训练数据，可以做思维链。
  * Codex论文中的代码数据量为 159G ，大约是初代 GPT-3 5700 亿训练数据的28%。code-davinci-002 及其后续变体可以做思维链推理。
  * 在 HELM 测试中，Liang et al. (2022) 对不同模型进行了大规模评估。 他们发现了针对代码训练的模型具有很强的语言推理能力，包括 120亿参数的code-cushman-001.。
  * 我们在 AI2 的工作也表明，当配备复杂的思维链时，code-davinci-002 在 GSM8K 等重要数学基准上是目前表现最好的模型
  * <mark style="background-color:green;">直觉来说，</mark><mark style="background-color:green;">**面向过程的编程 (procedure-oriented programming)**</mark> <mark style="background-color:green;"></mark><mark style="background-color:green;">跟人类</mark><mark style="background-color:green;">**逐步解决任务**</mark><mark style="background-color:green;">的过程很类似，</mark><mark style="background-color:green;">**面向对象编程 (object-oriented programming)**</mark> <mark style="background-color:green;"></mark><mark style="background-color:green;">跟人类</mark><mark style="background-color:green;">**将复杂任务分解为多个简单任务**</mark><mark style="background-color:green;">的过程很类似。</mark>
  * <mark style="background-color:red;">以上所有观察结果都是代码与推理能力 / 思维链 之间的相关性，但不一定是因果性。这种相关性很有趣，但现在还是一个待研究的开放性问题。目前看来，我们</mark><mark style="background-color:red;">**没有非常确凿的证据证明代码就是思维链和复杂推理的原因**</mark><mark style="background-color:red;">。</mark>
* 此外， **代码训练**另一个可能的副产品是长距离依赖，正如Peter Liu所指出：“语言中的下个词语预测通常是非常局部的，而代码通常需要更长的依赖关系来做一些事情，比如前后括号的匹配或引用远处的函数定义”。这里我想进一步补充的是：由于面向对象编程中的类继承，代码也可能有助于模型建立编码层次结构的能力。我们将对这一假设的检验留给未来的工作。



Where do these abilities come from?

Compared to the previous models, the two major differences are **instruction tuning** and **training on code**. Specifically

* The ability to **respond to** human **instructions** is a direct product of **instruction tuning**.
* The ability of **generalization** to **unseen instructions** is a **free lunch** given by **scaling** types of **instructions**, as is further evidenced by T0, Flan, and FlanPaLM papers
* The ability of **complex reasoning** with **chain-of-thought** is likely to be **a magical side product** of **training on code**:
  * The initial GPT-3 is not trained on code, and it cannot do chain-of-thought
  * The text-davinci-001, although being instruction tuned, ~~cannot do CoT~~ (corrected by Denny Zhou) can do CoT but the performance is significantly worse, as is reported by the first version of the CoT paper — so **instruction tuning may not be the reason for CoT. This leaves training on code to be be the number one suspect**.
  * PaLM has 5% code training data, and it can do chain-of-thought.
  * The code data in the codex paper is 159G, approximately 28% of the initial GPT-3 570G training data. code-davinci-002 and its subsequent variants can do chain-of-thought.
  * On the HELM evaluation, a massive-scale evaluation performed by Liang et al. (2022), the authors also found that models trained on/ for code has strong language reasoning abilities, including the 12B-sized code-cushman-001.
  * Our work at AI2 also shows that when equipped with complex chains of thought, code-davinci-002 is the SOTA model on important math benchmarks like GSM8K
  * As an intuition, think about how **procedure-oriented programming** is similar to **solving tasks step by step**, and how **object-oriented programming** is similar to **decomposing complex tasks into simpler ones**.
  * All the above observations are correlations between code and reasoning ability/ CoT. Such a correlation between code and reasoning ability/ CoT is very intriguing to the community and not well-understood. However, **there is still no hard evidence showing training on code is absolutely the reason for CoT and complex reasoning**. The source of CoT is still an open research problem.
* Additionally, **long-term dependency** might also be a nice side effect of **training on code.** As is pointed out by Peter Liu. “Next token prediction for language is usually very local, whereas code often requires longer dependencies to do things like close brackets or refer to distant defs”. I would further add: code may also give the model of encoding hierarchy due to inheritance in object-oriented programming. We leave the test of this hypothesis to future work.

另外还要注意一些细节差异：

* **text-davinci-002 与 code-davinci-002**
  * Code-davinci-002 是基础模型，text-davinci-002 是指令微调 code-davinci-002 的产物（见 OpenAI 的文档）。它在以下数据上作了微调：（一）人工标注的指令和期待的输出；（二）由人工标注者选择的模型输出。
  * 当有上下文示例 (in-context example) 的时候， Code-davinci-002 更擅长上下文学习；当没有上下文示例 / 零样本的时候， text-davinci-002 在零样本任务完成方面表现更好。从这个意义上说，text-davinci-002 更符合人类的期待（因为对一个任务写上下文示例可能会比较麻烦）。
  * OpenAI 不太可能故意牺牲了上下文学习的能力换取零样本能力 —— 上下文学习能力的降低更多是指令学习的一个副作用，OpenAI 管这叫对齐税。
* **001 模型（code-cushman-001 和 text-davinci-001）v.s. 002 模型（code-davinci-002 和 text-davinci-002）**
  * 001 模型主要是为了做纯代码 / 纯文本任务； 002 模型则深度融合了代码训练和指令微调，代码和文本都行。
  * Code-davinci-002 可能是第一个深度融合了代码训练和指令微调的模型。证据有：code-cushman-001 可以进行推理但在纯文本上表现不佳，text-davinci-001 在纯文本上表现不错但在推理上不大行。 code-davinci-002 则可以同时做到这两点。

There are certain detailed differences we would like to note:

* **text-davinci-002 v.s. code-davinci-002**
  * Code-davinci-002 is the base model, text-davinci-002 is the product of fine-tuning code-davinci-002 on (see documentation): (a). Human-annotated instructions and completions; (b). Self-generated completions chosen by human-annotators
  * Code-davinci-002 is better at in-context learning (when there are few task demonstrations); text-davinci-002 is better at zero-shot task completion (no demonstrations). In this sense, text-davinci-002 is more aligned with humans (because coming up with a task demonstration can be troublesome).
  * It is unlikely that OpenAI intentionally trades in-context learning ability for zero-shot ability — this tradeoff is more like a side effect of supervised instruction tuning (the alignment tax).
* **001 models (code-cushman-001 and text-davinci-001) v.s. 002 models (code-davinci-002 and text-davinci-002)**
  * The 001 models might be trained for code-~~only~~mainly / text-~~only~~mainly purpose (but still a mixture of text and code); the 002 models combines code tuning and instruction tuning
  * The first model after the combination is likely to be code-davinci-002. The supporting facts are that code-cushman-001 can do reasoning but not well on pure text, text-davinci-001 can do pure text but not well on reasoning. code-davinci-002 can do both.

#### 3.2 **这些能力是在预训练之后已经存在还是在之后通过微调注入？**

在这个阶段，我们已经确定了指令微调和代码训练的关键作用。一个重要的问题是如何进一步分析代码训练和指令微调的影响？具体来说：&#x20;

<mark style="background-color:orange;">上述三种能力是否</mark><mark style="background-color:orange;">**已经存在于初代的GPT-3**</mark><mark style="background-color:orange;">中，只是</mark><mark style="background-color:orange;">**通过指令和代码训练触发 / 解锁**</mark><mark style="background-color:orange;">？</mark>&#x20;

或者这些能力在初代的 GPT-3 中**并不存在**，是通过指令和代码训练**注入？**&#x20;

如果答案已经在初代的 GPT-3 中，**那么这些能力也应该在 OPT 中。 因此，要复现这些能力，或许可以直接通过指令和代码调整 OPT。** 但是，code-davinci-002 也可能不是基于最初的 GPT-3 davinci，而是基于比初代 GPT-3 更大的模型。如果是这种情况，可能就没办法通过调整 OPT 来复现了。研究社区需要进一步弄清楚 OpenAI 训练了什么样的模型作为 code-davinci-002 的基础模型。

我们有以下的假设和证据：

* code-davinci-002的**基础模型可能不是初代GPT-3 davinci 模型**。以下是证据：
  * 初代的GPT-3在数据集 C4 2016 - 2019 上训练，而 code-davinci-002 训练集则在延长到2021年才结束。因此 code-davinci-002 有可能在 C4 的 2019-2021 版本上训练。
  * 初代的 GPT-3 有一个大小为 **2048** 个词的上下文窗口。code-davinci-002 的上下文窗口则为 **8192**。GPT 系列使用绝对位置嵌入 (absolute positional embedding)，直接对绝对位置嵌入进行外推而不经过训练是比较难的，并且会严重损害模型的性能（参考 Press et al., 2022）。如果 code-davinci-002 是基于初代GPT-3，那OpenAI 是如何扩展上下文窗口的？
* 另一方面，无论基础模型是初代的 GPT-3 还是后来训练的模型， **遵循指令和零样本泛化的能力都可能已经存在于基础模型**中，后来才通过指令微调来**解锁** （**而不是注入）**
  * 这主要是因为 OpenAI 的论文报告的指令数据量大小只有 77K，比预训练数据少了几个数量级。
  * 其他指令微调论文进一步证明了数据集大小对模型性能的对比，例如 Chung et al. (2022) 的工作中， Flan-PaLM 的指令微调仅为预训练计算的 0.4%。一般来说，指令数据会显著少于预训练数据。
* 然而 **，模型的复杂推理能力可能是在预训练阶段通过代码数据注入**
  * 代码数据集的规模与上述指令微调的情况不同。这里的代码数据量足够大，可以占据训练数据的重要部分（例如，PaLM 有 5% 的代码训练数据）
  * 如上所述，在 code-davinci-002 之前的模型 text-davinci-001 大概没有在代码数据上面微调过，所以它的推理 / 思维链能力是非常差的，正如第一版思维链论文中所报告的那样，有时甚至比参数量更小的 code-cushman-001 还差。
*   <mark style="background-color:red;">**区分代码训练和指令微调效果的最好方法**</mark><mark style="background-color:red;">可能是</mark><mark style="background-color:red;">**比较 code-cushman-001、T5 和 FlanT5**</mark>

    * 因为它们具有相似的模型大小（110亿 和 120亿），相似的训练数据集 (C4)，它们最大的区别就是有没有在代码上训练过 / 有没有做过指令微调。
    * <mark style="background-color:red;">目前还没有这样的比较。我们把这个留给未来的研究。</mark>



At this stage, we already have identified the crucial role of instruction tuning and training on code. One important question is how to further disentangle the effects of code training and instruction tuning. Specifically:

**Are the above three abilities already there in the initial GPT-3** but **triggered/ unlocked by instruction and code training** or **not in the initial GPT-3** but **injected by instruction and code training?**

If the answer is already in the initial GPT-3, then these abilities **should also be in OPT**. **So to reproduce these abilities, one can directly instruction-and-code-tune OPT.** Yet it is also likely that code-davinci-002 is NOT based on the initial GPT-3 davinci, but some other models with unknown training procedures. If this is the case, tuning OPT might not be an option for reproduction, and the community need to figure out further what kind of model OpenAI has trained as the base model for code-davinci-002.

We have the following hypothesis and evidence:

* The **base model for code-davinci-002 is highly likely not be the initial GPT-3 davinci model**. Below are the evidence/ indicators
  * The initial GPT-3 is trained on C4 2016 - 2019. code-davinci-002 training set ends in 2021. So it is possible that code-davinci-002 is trained on the 2019-2021 version of C4.
  * The initial GPT-3 has a context window **2048**. code-davinci-002 has a context window ~~4096~~ **8192**. GPT series use absolute positional embedding, and directly extrapolating absolute positional embedding beyond training is challenging and can seriously harm the model performance (see Press et al., 2022). If code-davinci-002 is based on the initial GPT-3, how did OpenAI expand the context window?
  * There are recent works using sparse mixture-of-expert to substantially scale up the model parameter with constant computational cost, like Switch Transformers. If GPT-3.5 uses this technique, it can be **significantly larger** than GPT-3.
* On the other hand, either the base model is the initial GPT-3 or some later trained model, the ability to **follow instruction and zero-shot generalization may already be in the base model** and is later **unlocked** (**not injected**) by instruction tuning
  * This is primarily because the instruction data reported by OpenAI’s paper is only 77K, which is orders of magnitudes less than the pretraining data.
  * The contrast of dataset size is further evidenced by other instruction tuning papers, e.g., Chung et al. (2022) where instruction tuning of Flan-PaLM is just 0.4% compute of pretraining. Instruction data is generally significantly less than pretraining data.
* Yet **complex reasoning may be injected from code data during the pretraining stage**
  * The scale of code data is different than the above instruction tuning case. Here the amount of code data is large enough to take a nontrivial portion of the training data (e.g., PaLM has 8% code training data)
  * As mentioned above, the reasoning/ chain of thought ability in text-davinci-001, the model before code-davinci-002, presumably not tuned on code, is very low, as is reported in the first version of the chain-of-thought paper, sometimes even worse than a smaller code-cushman-001.
* The best way to **disentangle the effects of code tuning and instruction tuning** might be to **compare code-cushman-001, T5, and FlanT5**
  * Because they have similar size (11B and 12B), and similar training data (C4), and the only difference are code/ instruction tuning.
  * There are no such comparisons yet. We leave this to future research.

### 四、text-davinci-003 和 ChatGPT，基于人类反馈的强化学习(Reinforcement Learning from Human Feedback, RLHF) 的威力

<mark style="background-color:red;">在当前阶段（2022 年 12 月）， text-davinci-002、text-davinci-003 和 ChatGPT之间</mark><mark style="background-color:red;">**几乎没有严格的统计上的比较**</mark> <mark style="background-color:red;"></mark><mark style="background-color:red;">，主要是因为</mark>

* text-davinci-003 和 ChatGPT 在撰写本文时才发布不到一个月。
* ChatGPT 不能通过 OpenAI API 被调用，所以想要在标准基准上测试它很麻烦。

所以在这些模型之间的比较更多是**基于研究社区的集体经验** （统计上不是很严格）。不过，我们相信初步的描述性比较仍然可以揭示模型的机制。

At the current stage (Dec 2022), there are **few strict statistically clear comparisons** between text-davinci-002, text-davinci-003 and ChatGPT, mostly because

* text-davinci-003 and Chat have only been released less than a month when this article is written.
* ChatGPT cannot be called by OpenAI API, so it is troublesome to test it on standard benchmarks.

So the comparison between these models is more **based on the collective experiences of the community** (not very statistically strict). Yet we do believe certain initial descriptive comparisons still shed light on the underlying model mechanisms.

我们首先注意到以下 text-davinci-002，text-davinci-003 和 ChatGPT 之间的比较：

* 所有三个模型都经过**指令微调**。
* **text-davinci-002** 是一个经过**监督学习指令微调** (supervised instruction tuning) \*\*\*\*的模型
* **text-davinci-003 和 ChatGPT** 是**基于人类反馈的强化学习的指令微调** (Instruction tuning with Reinforcement Learning from Human Feedback, RLHF)。这是它们之间最显着的区别。

**这意味着大多数新模型的行为都是 RLHF 的产物**。

那么让我们看看 RLHF 触发的能力：

* **详实的回应：** text-davinci-003 的生成通常比 text-davinci-002长。 ChatGPT 的回应则更加冗长，以至于用户必须明确要求“用一句话回答我”，才能得到更加简洁的回答。这是 RLHF 的直接产物。
* **公正的回应**：ChatGPT 通常对涉及多个实体利益的事件（例如政治事件）给出非常平衡的回答。这也是RLHF的产物。
* **拒绝不当问题：**这是内容过滤器和由 RLHF 触发的模型自身能力的结合，过滤器过滤掉一部分，然后模型再拒绝一部分。
* **拒绝其知识范围之外的问题**：例如，拒绝在2021 年 6 月之后发生的新事件（因为它没在这之后的数据上训练过）。这是 RLHF 最神奇的部分，因为它使模型能够隐式地区分哪些问题在其知识范围内，哪些问题不在其知识范围内。

We first note that the following comparisons between text-davinci-002 v.s. text-davinci-003 v.s. ChatGPT:

* All three models are **instruction tuned**.
* t**ext-davinci-002** is a **supervised** instruction-tuned model
* t**ext-davinci-003 and ChatGPT** are instruction tuned with **Reinforcement Learning with Human Feedback (RLHF)**. This is the most prominent difference.

**This means that most of the new model behaviors are the product of RLHF**.

So let’s look at the abilities triggered by RLHF:

* **Informative responses:** text-davinci-003’s generation is usually longer than text-davinci-002. ChatGPT’s response is even more verbose such that one has to explicitly ask, “answer me in one sentence” to make it concise. This is a direct product of RLHF.
* **Impartial responses**: ChatGPT often gives very balanced responses on events involving interests from multiple entities, such as political events. This is also a product of RLHF
* \*\*Rejecting improper questions.\*\*This is the combination of a content filter and the model’s own ability induced by RLHF.
* **Rejecting questions outside its knowledge scope**: for example, rejecting new events that happened after Jun 2021. This is the most amazing part of RLHF because it enables the model to implicitly and automatically classify which information is within its knowledge and which is not.

有两件事情值得注意：

* 所有的能力都是模型本来就有的， **而不是通过RLHF 注入的**。 RLHF 的作用是**触发 / 解锁突现能力**。这个论点主要来自于数据量大小的比较：因为与预训练的数据量相比，RLHF 占用的计算量 / 数据量要少得多。
* 模型[**知道它不知道什么不是通过编写规则来实现的**](#user-content-fn-3)[^3]**，** 而是通过RLHF解锁的。这是一个非常令人惊讶的发现，因为 RLHF 的最初目标是让模型生成符合人类期望的回答，这更多是让模型生成安全的句子，而不是让模型知道它不知道的内容。

幕后发生的事情可能是：

* ChatGPT: 通过**牺牲上下文学习**的能力**换取建模对话历史**的能力。这是一个基于经验的观测结果，因为 ChatGPT 似乎不像 text-davinci-003 那样受到上下文演示的强烈影响。
* text-davinci-003：**恢复了** text-davinci-002 所牺牲的**上下文学习能力**， **提高零样本的能力**。 ~~我们不确定这是否也是 RLHF 或其他东西的副产品。~~ 根据instructGPT的论文，这是来自于强化学习调整阶段混入了语言建模的目标（而不是 RLHF 本身）。

There are two important things to notice:

* All the abilities are intrinsically within the model, **not injected by RLHF**. RLHF **triggers/unlock** these abilities to emerge. This is again because of the data size, as the RLHF tuning takes significantly less portion of computing compared to pretraining.
* **Knowing what it does not know is not achieved by writing rules;** it is also unlocked by RLHF. This is a very surprising finding, as the original goal of RLHF is for alignment, which is more related to generating safe responses than knowing what the model does not know.

What happens behind the scene might be:

* ChatGPT: **Trade in-context learning for dialog history modeling**. This is an empirical observation as ChatGPT seems not to be strongly affected by in-context demonstrations as text-davinci-003 does.
* Text-davinci-003: **recover the in-context learning ability** sacrificed by text-davinci-002 and **improve the zero-shot ability**. ~~We are not sure if this is also a side product of RLHF or something else.~~ According to the instructGPT paper, this is from the LM-mixing during the RL tuning stage (not RLHF itself).

### 五、总结**当前阶段 GPT-3.5 的进化历程**

到目前为止，我们已经仔细检查了沿着进化树出现的所有能力，下表总结了演化路径：

| 能力                                                                                                                                        | OpenAI模型                                            | 训练方法                | OpenAI API                            | OpenAI论文                            | 近似的开源模型                                      |
| ----------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------- | ------------------------------------- | ----------------------------------- | -------------------------------------------- |
| <mark style="background-color:yellow;">GPT-3系列</mark>                                                                                     |                                                     |                     |                                       |                                     |                                              |
| <p>语言生成</p><p>+ 世界知识 </p><p>+ 上下文学习</p>                                                                                                   | GPT-3初始版本  **大部分的能力已经存在于模型中，尽管表面上看起来很弱。**           | 语言建模                | Davinci                               | GPT-3论文                             | Meta OPT                                     |
| <p>+ 遵循人类的指令</p><p>+ 泛化到没有见过的任务</p>                                                                                                       | Instruct-GPT初始版本                                    | 指令微调                | Davinci-Instruct-Beta                 | Instruct-GPT论文                      | <p>T0论文 </p><p>Google FLAN论文</p>             |
| + 代码理解+ 代码生成                                                                                                                              | Codex初始版本                                           | 在代码上进行训练            | Code-Cushman-001                      | Codex论文                             | Salesforce CodeGen                           |
| <mark style="background-color:yellow;">GPT-3.5系列</mark>                                                                                   |                                                     |                     |                                       |                                     |                                              |
| <p>++ 代码理解 </p><p>++ 代码生成 </p><p>++ 复杂推理 / 思维链 (为什么?)</p><p>+ 长距离的依赖 (很可能)</p>                                                            | 现在的Codex**GPT3.5系列中最强大的模型**                         | 在代码+文本上进行训练在指令上进行微调 | Code-Davinci-002 (目前免费的版本 = 2022年12月) | Codex 论文                            |                                              |
| <p>++ 遵循人类指令</p><p>- 上下文学习</p><p>- 推理能力</p><p>++ 零样本生成</p>                                                                                | 有监督的Instruct-GPT **通过牺牲上下文学习换取零样本生成的能力**            | 监督学习版的指令微调          | Text-Davinci-002                      | Instruct-GPT论文, 有监督的部分              | <p>T0论文</p><p>Google FLAN论文</p>              |
| <p>+ 遵循人类价值观 </p><p>+ 包含更多细节的生成</p><p>+ 上下文学习</p><p>+ 零样本生成</p>                                                                           | 经过RLHF训练的Instruct-GPT**和002模型相比，和人类更加对齐，并且更少的性能损失** | 强化学习版的指令微调          | Text-Davinci-003                      | Instruct-GPT论文, RLHF部分，从人类反馈中的学习摘要。 | <p>DeepMind Sparrow 论文 </p><p>AI2 RL4LMs</p> |
| <p>++ 遵循人类价值观 ++ 包含更多细节的生成 </p><p>++ <mark style="background-color:red;">拒绝知识范围外的问题 (为什么?)</mark> </p><p>++ 建模对话历史的能力 </p><p>-- 上下文学习</p> | ChatGPT **通过牺牲上下文学习的能力取取建模对话历史的能力**                 | 使用对话数据进行强化学习指令微调    |                                       |                                     | <p>DeepMind Sparrow论文</p><p>AI2 RL4LMs</p>   |

我们可以得出结论：

* 语言生成能力 + 基础世界知识 + 上下文学习都是来自于预训练（`davinci`）
* 存储大量知识的能力来自 1750 亿的参数量。
* 遵循指令和泛化到新任务的能力来自于扩大指令学习中指令的数量（`Davinci-instruct-beta`)
* 执行复杂推理的能力很可能来自于代码训练（`code-davinci-002`）
* <mark style="background-color:green;">生成中立、客观的能力、安全和翔实的答案来自与人类的对齐。具体来说：</mark>
  * 如果是监督学习版，得到的模型是`text-davinci-002`
  * 如果是强化学习版 (RLHF) ，得到的模型是`text-davinci-003`
  * 无论是有监督还是 RLHF ，模型在很多任务的性能都无法超过 code-davinci-002 ，这种因为对齐而造成性能衰退的现象叫做对齐税。
* 对话能力也来自于 RLHF（`ChatGPT`），具体来说它牺牲了上下文学习的能力，来换取：
  * 建模对话历史
  * 增加对话信息量
  * 拒绝模型知识范围之外的问题

We have concluded:

* The language generation ability + basic world knowledge + in-context learning are from pretraining (`davinci`)
* The ability to store a large amount of knowledge is from the 175B scale.
* The ability to follow instructions and generalizing to new tasks are from scaling instruction tuning (`davinci-instruct-beta`)
* The ability to perform complex reasoning is likely to be from training on code (`code-davinci-002`)
* The ability to generate neutral, objective, safe, and informative answers are from alignment with human. Specifically:
  * If supervised tuning, the resulting model is `text-davinci-002`
  * If RLHF, the resulting model is `text-davinci-003`
  * Either supervised or RLHF, the models cannot outperform code-davinci-002 on many tasks, which is called the alignment tax.
* The dialog ability is also from RLHF (`ChatGPT`), specifically it tradeoffs in-context learning for:
  * Modeling dialog history
  * Increased informativeness
  * Rejecting questions outside the model’s knowledge scope



### 六、GPT-3.5 目前不能做什么

虽然GPT-3.5是自然语言处理研究中的重要一步，但它并没有完全包含许多研究人员（包括 AI2）设想的所有理想属性。以下是GPT-3.5不具备的某些重要属性：

* **实时改写模型的信念**：当模型表达对某事的信念时，如果该信念是错误的，我们可能很难纠正它：
  * 我最近遇到的一个例子是：ChatGPT 坚持认为 3599 是一个质数，尽管它承认 3599 = 59 \* 61。另外，请参阅Reddit上关于游得最快的海洋哺乳动物的例子。
  * 然而，模型信念的强度似乎存在不同的层次。一个例子是即使我告诉它达斯·维达（星球大战电影中的人物）赢得了2020年大选，模型依旧会认为美国现任总统是拜登。但是如果我将选举年份改为 2024 年，它就会认为总统是达斯·维达是 2026 年的总统。
* **形式推理**：GPT-3.5系列不能在数学或一阶逻辑等形式严格的系统中进行推理：
  * 在自然语言处理的文献中， “推理” 一词的定义很多时候不太明确。但如果我们从模糊性的角度来看，例如一些问题 (a) 非常模棱两可，没有推理；(b) 有点儿逻辑在里面，但有些地方也可以模糊；(c) 非常严谨，不能有任何歧义。那么，
  * 模型可以很好地进行 (b) 类的带模糊性的推理，例子有：
    * 生成如何做豆腐脑的方法。做豆腐脑的时候，中间很多步骤模糊一点是可以接受的，比如到底是做咸的还是做甜的。只要整体步骤大致正确，做出来的豆腐脑儿就能吃。
    * 数学定理的证明思路。证明思路是用语言表达的非正式的逐步解法，其中每一步的严格推导可以不用太具体。证明思路经常被用到数学教学：只要老师给一个大致正确的整体步骤，学生就可以大概明白。然后老师把具体的证明细节作为作业布置给学生，答案略。
  * GPT-3.5 不能进行类型 (c) 的推理（推理不能容忍歧义）。
    * 一个例子是严格的数学证明，要求中间步骤中不能跳，不能模糊，不能错。
    * 但这种严格推理到底是应该让语言模型做还是让符号系统做还有待讨论。一个例子是，与其努力让 GPT 做三位数加法，不如直接调 Python。
* **从互联网进行检索**：GPT-3.5 系列（暂时）不能直接搜索互联网
  * 但是有一篇 WebGPT 论文发表于2021年12月，里面就让 GPT 调用了搜索引擎。所以检索的能力已经在 OpenAI 内部进行了测试。
  * 这里需要区分的一点是，GPT-3.5 的两个重要但不同的能力是 **知识** 和 **推理**。一般来说，如果我们能够 **将知识部分卸载到外部的检索系统，让语言模型只专注于推理，这就很不错了。** 因为：
    * 模型的内部知识总是在某个时间被切断。模型始终需要最新的知识来回答最新的问题。
    * 回想一下，我们已经讨论过 1750 亿的参数大量用于存储知识。如果我们可以将知识卸载到模型之外，那么模型参数可能会大大减少，最终它甚至可以在手机上运行（疯狂的想法，但 ChatGPT 已经足够科幻了，谁知道未来会怎样呢).

Although it is a major step in NLP research, GPT-3.5 does not fully contain all the ideal properties envisaged by many NLP researchers (including AI2). Below are certain important properties that GPT-3.5 does not have:

* **On-the-fly overwriting the model’s belief**: when the model expresses its belief in something, it might be hard to correct it when the belief is wrong:
  * One recent example I encountered is that ChatGPT insists that 3599 is a prime number even though it acknowledged that 3599 = 59 \* 61. Also, see the fastest marine mammal example on Reddit.
  * Yet there seems to be a hierarchy of how strong the belief is. One example is that the model believes the current president of the US is Biden, even if I told it Darth Vader won the 2020 election. Yet if I change the election year to 2024, it believes that the president is Darth Vader in 2026.
* **Formal reasoning**: the GPT-3.5 series cannot do reasoning within formal, strict systems like math or first-order logic
  * In the NLP literature, the word “reasoning” is less well-defined. Yet if we view there is a spectrum of ambiguity like (a) very ambiguous, no reasoning; (b) mixture of logic and ambiguous statements; (c). no ambiguity has to be very rigorous, then,
  * The model can do very well on type (b) reasoning with ambiguity; examples include:
    * Generating a procedure of how to cook pizza. It is acceptable if there exist ambiguities in the intermediate steps, like using sausage or pineapple. As long as the overall steps are approximately correct, the pizza is eatable (sorry if you are Italian).
    * Generating proof sketch of a theorem. Proof sketches are informal step-by-step procedures expressed in language, where the strict derivation of one step can be left unspecified. This is a useful math teaching and thinking tool. As long as the overall steps are approximately correct, the students can fill in the details as homework.
  * The model cannot do type (c) reasoning (reasoning does not tolerate ambiguity).
    * One example is deriving strict proofs that require no mistakes in intermediate steps.
    * Yet whether such reasoning should be done by a language model or a symbolic system is up for discussion. For example, instead of trying hard to make GPT do three digits addition, one might simply call Python.
* **Retrieval from the Internet**: the GPT-3.5 series cannot directly search the internet (for now)
  * Yet there was a WebGPT paper published in Dec 2021. It is likely that this is already tested internally within OpenAI.
  * The two important but different abilities of GPT-3.5 are **knowledge** and **reasoning**. Generally, it would be ideal if we could **offload the knowledge part to the outside retrieval system and let the language model only focus on reasoning.** This is because:
    * The model’s internal knowledge is always cut off at a certain time. The model always needs up-to-date knowledge to answer up-to-date questions.
    * Recall we have discussed that is 175B parameter is heavily used for storing knowledge. If we could offload knowledge to be outside the model, then the model parameter might be significantly reduced such that eventually, it can run on a cellphone (call this crazy here, but ChatGPT is already science fiction enough, who knows what the future will be).

## 七、结论

在这篇博文中，我们仔细检查了GPT-3.5系列的能力范围，并追溯了它们所有突现能力的来源。初代GPT-3模型通过预训练获得生成能力、世界知识和in-context learning。然后通过instruction tuning的模型分支获得了遵循指令和能泛化到没有见过的任务的能力。经过代码训练的分支模型则获得了代码理解的能力，作为代码训练的副产品，模型同时潜在地获得了复杂推理的能力。结合这两个分支，code-davinci-002似乎是具有所有强大能力的最强GPT-3.5模型。接下来通过有监督的instruction tuning和 RLHF通过牺牲模型能力换取与人类对齐，即对齐税。 RLHF 使模型能够生成更翔实和公正的答案，同时拒绝其知识范围之外的问题。

我们希望这篇文章能够帮助提供一个清晰的GPT评估图，并引发一些关于语言模型、instruction tuning和code tuning的讨论。最重要的是， **我们希望这篇文章可以作为在开源社区内复现GPT-3.5的路线图。**

> “因为山就在那里。”——乔治·马洛里，珠穆朗玛峰探险先驱

In this post, we scrutinize the spectrum of abilities of the GPT-3.5 series and trace back to the sources of all their emergent abilities. The initial GPT-3 model gains its generation ability, world knowledge, and in-context learning from pretraining. Then the instruction tuning branch gains the ability to follow instructions and generalization to unseen tasks. The training on code branch gains the ability of code understanding, and potentially the side product of complex reasoning. Combining the two branches, code-davinci-002 seems to be the most capable GPT-3.5 model with all the powerful abilities. The following supervised instruction tuning and RLHF trades model ability for alignment with humans, i.e., the alignment tax. RLHF enables the model to generate more informative and impartial answers while rejecting questions outside its knowledge scope.

We hope this article can help provide a clear picture of the evaluation of GPT, and stir some discussion about language models, instruction tuning, and code tuning. Most importantly, **we hope this article can serve as the roadmap for reproducing GPT-3.5 within the open-source community.**

> “Because it's there” — George Mallory, the pioneer of the Mount Everest expedition

## **常见问题**

* 这篇文章中的这些说法更像是假设 (hypothesis) 还是结论 (conclusion)？
  * **复杂推理的能力来自于代码训练**是我们倾向于相信的假设 (hypothesis)
  * **对没有见过的任务泛化能力来自大规模指令学习** 是至少 4 篇论文的结论 (conclusion)
  * **GPT-3.5来自于其他大型基础模型，而不是1750亿参数的GPT-3** 是有根据的猜测 (educated guess)。
  * **所有这些能力都已经存在了，通过instruction tuning，无论是有监督学习或强化学习的方式来解锁而不是注入这些能力** 是一个比较强的假设 (strong assumption)。 主要是因为instruction tuning数据量比预训练数据量少了几个数量级。
  * 结论 (conclusion) = 许多证据支持这些说法的正确性；假设 (hypothesis) = 有正面证据但不够有力；有根据的猜测 (educated guess) = 没有确凿的证据，但某些因素会指向这个方向
*   为什么其他模型（如 OPT 和 BLOOM）没有那么强大？

    * OPT大概是因为训练过程太不稳定
    * BLOOM的情况则未知。如果您有更多意见，请与我联系


* Are these claims in this article more like hypothesis or conclusions?
  * **Complex reasoning is from training on code** is a hypothesis we tend to believe
  * **Generalization on unseen tasks is from scaling instruction tuning** is a conclusion from at least 4 papers
  * **GPT-3.5 is from a large base model than GPT-3 175B** is an educated guess.
  * **All these abilities are there, instruction tuning, either supervised or reinforce, unlocks, but not inject, these abilities** is a strong hypothesis, very strong such that it is hard not to believe. Mostly because instruction tuning data are orders of magnitudes less than pretraining data
  * Conclusion = many evidences supporting its correctness; Hypothesis = there are positive evidences but not strong enough; Educated guess = no hard evidence but certain factors indicate so
* Why other models like OPT and BLOOM are not so strong?
  * OPT probably because training process too unstable
  * BLOOM do not know. Do contact me if you have more comments

## 附录 - 中英术语对照表

| 英文                                                | 中文          | 释义                                     |
| ------------------------------------------------- | ----------- | -------------------------------------- |
| Emergent Ability                                  | 突现能力        | 小模型没有，只在模型大到一定程度才会出现的能力                |
| Prompt                                            | 提示词         | 把 prompt 输入给大模型，大模型给出 completion       |
| In-Context Learning                               | 上下文学习       | 在 prompt 里面写几个例子，模型就可以照着这些例子做生成        |
| Instruction Tuning                                | 指令微调        | 用 instruction 来 fine-tune 大模型          |
| Code Tuning                                       | 在代码上微调      | 用代码来 fine-tune 大模型                     |
| Reinforcement Learning with Human Feedback (RLHF) | 基于人类反馈的强化学习 | 让人给模型生成的结果打分，用人打的分来调整模型                |
| Chain-of-Thought                                  | 思维链         | 在写 prompt 的时候，不仅给出结果，还要一步一步地写结果是怎么推出来的 |
| Scaling Laws                                      | 缩放法则        | 模型的效果的线性增长要求模型的大小指数增长                  |
| Alignment                                         | 与人类对齐       | 让机器生成符合人类期望的，符合人类价值观的句子                |

[^1]: * &#x20;[https://thegradient.pub/in-context-learn](https://thegradient.pub/in-context-learning-in-context/)

    <!---->

    * [Min et. al. 2022. Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/abs/2202.12837)
    * [An Explanation of In-context Learning as Implicit Bayesian Inference](https://arxiv.org/abs/2111.02080)

    <!---->

    *

    <img src="https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b9f99cde-06ab-473c-b67d-f08447a7f3ce/Untitled.png" alt="Untitled" data-size="original">













[^2]: ?



[^3]: GPT4的解释：对于一个系统（如人工智能或个体），要意识到自己所不知道的信息或知识，并非通过制定一系列规则来达到这一目标。换句话说，通过规则编写可能无法使一个系统完全了解自己的知识边界和不足之处。实现这一目标可能需要更为复杂的方法，如动态学习和自我调整。
