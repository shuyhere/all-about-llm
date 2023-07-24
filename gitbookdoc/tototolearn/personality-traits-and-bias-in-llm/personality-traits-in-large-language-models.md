---
description: >-
  "Present a comprehensive method for administering validated psychometric tests
  and quantifying, analyzing, and shaping personality traits exhibited in text
  generated from widely-used LLMs."
---

# 👹 Personality Traits in Large Language Models

## Abstract

Since personality is an important factor determining the effectiveness of communication, we present a comprehensive method for administering validated psychometric tests and quantifying, analyzing, and shaping personality traits exhibited in text generated from widely-used LLMs. We find that: 1) personality simulated in the outputs of some LLMs (under specific prompting configurations) is reliable and valid; 2) evidence of reliability and validity of LLM-simulated personality is stronger for larger and instruction fine-tuned models; and 3) personality in LLM outputs can be shaped along desired dimensions to mimic specific personality profiles. We also discuss potential applications and ethical implications of our measurement and shaping framework, especially regarding responsible use of LLMs.

主要发现：

1）一些LLM的输出中模拟的人格（在特定的prompt配置下）是可靠和有效的；&#x20;

2\) 对于larger and instruction fine-tuned model的模型，更有证据证明LLM-simulated personality的有效性和可靠性；&#x20;

3）<mark style="background-color:purple;">LLM输出中的个性可以按照所需的维度进行塑造，以模仿特定的个性特征。</mark>



## 主要内容

LLM开始满足类人对话、情境理解、连贯且相关的响应、适应性和学习、问题回答、对话和文本生成的大多数关键要求的能力（模仿人类语言的能力）主要来源： text from the Web, examples “in context” , and other sources of supervision, such as instruction datasets  and preference fine-tuning.

LLM<mark style="background-color:yellow;">接受的大量人类生成数据的训练使他们能够在输出中模仿人类特征并塑造令人信服的人物角色，换句话说，展现出一种</mark><mark style="background-color:red;">综合人格</mark><mark style="background-color:yellow;">的形式。</mark>

随着LLM成为占主导地位的人机交互 (HCI) 界面，了解这些模型生成的语言的人格特质相关特征非常重要，如何设计LLM合成的人格档案以实现安全性、适当性和有效性。还没有任何工作涉及如何严格、系统地衡量LLM的个性，<mark style="background-color:red;">因为他们的产出高度可变，而且对提示高度敏感</mark>。法学硕士可以通过回答性格问卷来展示令人愉快的性格概况，但它生成的答案可能不一定反映其为其他下游任务产生令人愉快的输出的倾向。例如，当在客户服务环境中部署为对话式聊天机器人时，同一个LLM也可能会猛烈地斥责客户。

**Our work aims to answer:**

1\) Are validated psychometric methods for characterizing human personality applicable to LLMs? 于表征人类性格的经过验证的心理测量方法是否适用于LLM？

2\) After applying validated psychometrics, does LLM-generated language exhibit personality traits in valid, reliable and meaningful ways similar to human-generated language? 在应用经过验证的心理测量学之后，LLM生成的语言是否以类似于人类生成的语言的有效、可靠和有意义的方式表现出人格特征？

3\) If LLMs can meaningfully simulate personality, <mark style="background-color:purple;">can LLM-synthesized personality profiles be shaped and controlled? LLM合成的人格可以塑造和控制吗？</mark>



<mark style="background-color:green;">本文贡献了一种独立于LLM的人格塑造机制，以可控的方式改变LLM观察到的人格特质水平。</mark>





