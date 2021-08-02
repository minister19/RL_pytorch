# Intro

This repo records Shuang Gao's AI learning progress.

### Ask for leaves.

2021-04-29 Shawn: ask for leave. OT in ABB, cannot get home.
2021-05-02 Shawn: ask for leave. Travelling to Tangyin.
2021-05-03 Shawn: ask for leave. Travelling to Tangyin.
2021-05-13 Shawn: ask for leave. Play SC2.
2021-05-13 Shawn: ask for leave. Xiaoyun visits.

# TODO

1. Research that tuning reward such that q table reaches convergence.
   phenomenon: maze_1d, for Q Learning, Q(state=1, action=0) === -2.0
2. 比较线性网络和卷积神经网络。因为信号的曲线可以看作是一种图形。
3. Zigzag 指标做数据库标记，LSTM 模型做状态解析（利用它可以分析一段数据的特点），训练出状态-决策模型。

# Futures_ctp

### Futures_ctp NOTES

1. How comes feedback signal? 每个信号发生以后，还应持续观察从此以后的价差，用于判断之前的信号是否合理.
2. Important assumptions:

- Sig-feedback is independent of sig
- Sig's feedback is reset on each new sig, regardless of sign
- Margin is recorded on each new action

### Futures_ctp TODO

1. 2020-08-18 Shawn: punish frequent trade.
2. 2020-08-18 Shawn: 仅当 reward 较大时保存 memory.
3. 2020-08-18 Shawn: 打印 reward 曲线，验证指标.
