# 强化学习相关
## RW

### 系列1
[OpenRLHF源码解读：理解Reward Model训练过程](https://zhuanlan.zhihu.com/p/14993645091)

[聊聊PRM（过程奖励模型）](https://zhuanlan.zhihu.com/p/15540962086)

[OpenRLHF源码解读：理解PRM(过程奖励模型)训练过程](https://zhuanlan.zhihu.com/p/16027048017)

### 系列2
[【RLHF】想训练ChatGPT？先来看看强化学习（RL）+语言模型（LM）吧（附源码）](https://zhuanlan.zhihu.com/p/606328992)

[【RLHF】想训练ChatGPT？得先弄明白Reward Model怎么训（附源码）](https://zhuanlan.zhihu.com/p/595579042)

## Loss计算
loss计算涉及chosen_reward和rejected_reward，计算逻辑都是在基础模型（如sft模型）的最后一层，套一个分类层：nn.Linear(config.hidden_size, 1, bias=False)，将token的隐藏层转化成一个打分值，基于这个打分值来计算chosen_reward、rejected_reward，再根据-torch.nn.functional.logsigmoid(chosen_reward - rejected_reward)计算loss

1.可以参考：

[文档](https://zhuanlan.zhihu.com/p/14993645091)
[代码](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/model.py)

2.也可以参考：

[文档](https://zhuanlan.zhihu.com/p/4535749790)
[代码](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/model/reward_model.py)

## PPO

### 相关论文
[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)
### 系列1
[OpenRLHF源码解读：1.理解PPO单机训练](https://zhuanlan.zhihu.com/p/13043187674)

[OpenRLHF源码解读：2.PPO训练Experience数据采样过程](https://zhuanlan.zhihu.com/p/14569025663)

[OpenRLHF源码解读：3.PPO模型训练过程](https://zhuanlan.zhihu.com/p/14813158239)

### 系列2
[图解大模型RLHF系列之：人人都能看懂的PPO原理与源码解读](https://zhuanlan.zhihu.com/p/677607581)

[RLHF通俗理解](https://zhuanlan.zhihu.com/p/685261886)

## DPO

### 系列1
[DPO 是如何简化 RLHF 的](https://zhuanlan.zhihu.com/p/671780768)

[DPO: Direct Preference Optimization 直接偏好优化（学习笔记）](https://www.cnblogs.com/lemonzhang/p/17910358.html)

[DPO: Direct Preference Optimization 论文解读及代码实践](https://zhuanlan.zhihu.com/p/642569664)

[使用 DPO 微调 Llama 2](https://huggingface.co/blog/zh/dpo-trl)

[监督式微调(SFT) & 偏好对齐(DPO)：From Zero To Hero](https://zhuanlan.zhihu.com/p/715250294)

## GRPO

## RFT

## 综合
### 系列1
[大模型是这样炼成的](https://mp.weixin.qq.com/s/D6gtKpNm9PP-NCm2PdUFlg)

[大模型中的强化学习——大语言模型研究05](https://limoncc.com/post/c0a3be9c86b2b4cd/)

[一文搞懂大模型强化学习策略：DPO、PPO和GRPO](https://mp.weixin.qq.com/s/JKpkgGHqyAVG95sGC1JFDQ)

# Agent相关

## Context Engineering
[让manus从零到一的上下文工程到底是什么？一文起底](https://mp.weixin.qq.com/s/olCKdXCNKuu1nnjaUWHFvA)

[万字长文深入浅出教你优雅开发复杂AI Agent](https://mp.weixin.qq.com/s/eon4MCCErRWLT7GxSoR70g?poc_token=HGFff2ij1drPeQOv1I219g43eOnLonQtjC60hJI9)

## 智能体框架
- MetaGPT
  - [官网](https://www.deepwisdom.ai/)
  - [github](https://github.com/FoundationAgents/MetaGPT)
  - [文档](https://docs.deepwisdom.ai/main/zh/)
- OpenManus
  - [官网](https://openmanus.org/)
  - [github](https://github.com/FoundationAgents/OpenManus)
- AutoGen
  - [官网](https://microsoft.github.io/autogen/stable/)
  - [github](https://github.com/microsoft/autogen)
  - [文档](https://microsoft.github.io/autogen/0.2/docs/Getting-Started/)
- AutoGPT
  - [官网](https://autogpt.net/)
  - [github](https://github.com/Significant-Gravitas/AutoGPT)
  - [文档](https://docs.agpt.co/)
- 字节deep search
  - [github](https://github.com/volcengine/ai-app-lab/tree/main/demohouse/deep_search_mcp/backend)


