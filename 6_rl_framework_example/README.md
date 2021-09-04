# Futures_ctp

### Futures_ctp NOTES

1. How comes feedback signal? 每个信号发生以后，还应持续观察从此以后的价差，用于判断之前的信号是否合理.
2. Important assumptions:

- Sig-feedback is independent of sig.
- Sig's feedback is reset on each new reverse sig.
- Margin is recorded on each new action.
- Fund is independent of model.

### Futures_ctp TODO

1. 优化 action 在 Kline 上的显示
2. action 切换超过一定数量即结算 episode，用以约束 action 频繁切换
3. 不同周期数据，通过插值合并到统一训练集内
4. 不同周期信号的 tick 比较重要，或者短窗口 tick，用于极端行情的风控
5. LSTM 模型做状态解析（利用它可以分析一段数据的特点），Zigzag 指标做数据库标记，训练出有监督状态-决策模型。
6. LSTM 模型做状态解析（利用它可以分析一段数据的特点），Q learning，训练出无监督状态-决策模型。Same idea on line: https://medium.com/@Lidinwise/trading-through-reinforcement-learning-using-lstm-neural-networks-6ffbb1f5e4a5
7. update target model only when fund_totals curve increases.

### Research Notes

1. Verified that margins are calculated correctly, using excel.
2. Change action to markers.

### Archived ideas

1. 2020-08-18 Shawn: 仅当 reward 绝对值较大时保存 memory. Reason: 不能有幸存者偏差，走向局部优化。
2. 2020-08-18 Shawn: 打印 reward 曲线，验证指标. Reason: Done.
3. 2021-08-29 Shawn: 'U' action means to hold on to previous action. Reason: Well this action can be saved if reduce action transitions.
4. 2020-08-18 Shawn: design algorithm to reduce action transitions. Reason: Done.
5. reward = 1 + margin, 1 for if margin >=0, one step forward. Reason: trade_fee has the same effect.
