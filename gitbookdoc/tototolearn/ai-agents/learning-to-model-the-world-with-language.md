---
description: Dynalang 利用多种类型的语言来解决任务，通过使用语言来预测多模式世界模型中的未来。
---

# 🍞 Learning to Model the World with Language

{% embed url="https://dynalang.github.io/" %}

[paper](https://arxiv.org/abs/2308.01399)

[code](https://github.com/jlin816/dynalang) 还没有更新

为了与人类互动并在世界中行动，智能体需要了解人们使用的语言范围并将其与视觉世界联系起来。 虽然<mark style="background-color:red;">当前的智能体通过任务奖励学习执行简单的语言指令</mark>，但我们的目标是构建利用多种语言来传达常识、描述世界状态、提供交互式反馈等的智能体。&#x20;

我们的关键想法是，<mark style="background-color:red;">语言可以帮助智能体预测未来：</mark>**将观察到什么，世界将如何表现，以及哪些情况将得到奖励。**&#x20;

<mark style="background-color:green;">这种观点将</mark><mark style="background-color:green;">**语言理解与未来预测**</mark><mark style="background-color:green;">结合起来，作为一个强大的自我监督学习目标。</mark> 我们提出了 Dynalang，这是一种学习多模态世界模型的代理，该模型可以预测未来的文本和图像表示，并学习从想象的模型展示中采取行动。 与仅使用语言来预测动作的传统代理不同，Dynalang 通过使用过去的语言来预测未来的语言、视频和奖励，从而获得丰富的语言理解。 除了从环境中的在线交互中学习之外，Dynalang 还可以在文本、视频或两者的数据集上进行预训练，而无需操作或奖励。 从在网格世界中使用语言提示到导航房屋的逼真扫描，Dynalang 利用不同类型的语言来提高任务性能，包括环境描述、游戏规则和说明。
