# Futures_ctp

### Futures_ctp NOTES

1. How comes feedback signal? 每个信号发生以后，还应持续观察从此以后的价差，用于判断之前的信号是否合理.
2. Important assumptions:

- Sig-feedback is independent of sig.
- Sig's feedback is reset on each new sig, regardless of sign.
- Margin is recorded on each new action.
- Assume Fund, Position, Margin be independent of model.

### Futures_ctp TODO

1. 2020-08-18 Shawn: punish frequent trade.
2. 2020-08-18 Shawn: 仅当 reward 较大时保存 memory.
3. 2020-08-18 Shawn: 打印 reward 曲线，验证指标.
4. Zigzag 指标做数据库标记，LSTM 模型做状态解析（利用它可以分析一段数据的特点），训练出有监督状态-决策模型。
