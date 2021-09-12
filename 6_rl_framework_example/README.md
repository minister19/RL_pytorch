# Futures_ctp

### Futures_ctp Model

1. How comes feedback signal? 每个信号发生以后，还应持续观察从此以后的价差，用于判断之前的信号是否合理.
2. Important assumptions:

- Sig-feedback is independent of sig.
- Sig's feedback is reset on each new reverse sig.
- Margin is recorded on each new action.
- Fund is independent of model.

### Futures_ctp Roadmap

1. Verified that margins are calculated correctly, using excel.
2. Design algorithm to reduce action transitions: trade_fee algorithm (improved from absolute-based to open/close_vol-based).
3. Plot klines and margins.
4. Plot actions as markers.
5. Margin basing on avg_cost rather than margin_base, otherwise model would not work for general purpose.
6. Consider action 'N', if action is 'N', reward is small value wrt trade_fee. 优点：较小的回撤。缺点：收益变低，误操作也有。
7. Be aware if training more than enough times, model is overfitting (loss drops, rise and drop again).
8. Evaluate how (close) have our indicators revealed nature of money/fund flow in the market, given a fixed period.

### Archived ideas

1. 2020-08-18 Shawn: 仅当 reward 绝对值较大时保存 memory. Reason: 不能有幸存者偏差，走向局部优化。
2. 2021-08-29 Shawn: 'U' action means to hold on to previous action. Reason: saved by trade_fee algorithm.
3. 2021-08-31 Shawn: reward = 1 + margin, 1 for if margin >=0, 1 step forward. Reason: saved by trade_fee algorithm.
4. 2021-09-05 Shawn: If fixed number of actions are used up, the last action's state is calculated by last kline rather than next kline. Action 切换超过一定数量即结算 episode，用以约束 action 频繁切换: early-done algorithm. Reason: trainning data (klines) are also reduced, deteriorates learning progress, should be useful as part of risk management module.
5. 2021-09-05 Shawn: For every x steps, only y actions can be taken, if exceeds, done this episode. Reason: variant of early-done algorithm.
6. 2021-09-06 Shawn: update target model only when fund_totals curve increases. Reason: 不能有幸存者偏差，走向局部优化。
7. If action alignes with withdraw signal, action applies tick price instead of close price to simulate tick operations. Reason: parctice in production mode, if kline finishes and action does not align with withdraw signal, close position.
8. 由于资金有限，实际生产环境时，无法对所有监测交易品进行全时段持有，所以要根据信号可靠性进一步筛选，间断持有。但某些高度关注/用于科研的交易品可以设置为全时段持有。

### TODO:

1. More Klines to train.

### Futures_ctp to-research

1. 不同周期数据，通过插值合并到统一训练集内
2. 不同周期信号的 tick 比较重要，或者短窗口 tick，用于极端行情的风控
3. LSTM 模型做状态解析（利用它可以分析一段数据的特点），Zigzag 指标做数据库标记，训练出有监督状态-决策模型。
4. LSTM 模型做状态解析（利用它可以分析一段数据的特点），Q learning，训练出无监督状态-决策模型。Same idea on line: https://medium.com/@Lidinwise/trading-through-reinforcement-learning-using-lstm-neural-networks-6ffbb1f5e4a5
5. Consider norminal margin as stepped ones, rather than decimal ones.
6. If fixed number of actions are used up, the last action's state is calculated by last kline rather than next kline. Action 切换超过一定数量即结算 episode，用以约束 action 频繁切换: early-done algorithm. 作为通用人工智能模型的一个模块。
