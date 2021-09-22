# Intro

This repo records Shuang Gao's AI learning progress.

### Ask for leaves.

2021-04-29 Shawn: ask for leave. OT in ABB, cannot get home.
2021-05-02 Shawn: ask for leave. Travelling to Tangyin.
2021-05-03 Shawn: ask for leave. Travelling to Tangyin.
2021-05-13 Shawn: ask for leave. Play SC2.
2021-05-13 Shawn: ask for leave. Xiaoyun visits.
2021-09-22 Shawn: ask for leave. Illness recovery.

# TODO

1. Research that tuning reward such that q table reaches convergence.
   phenomenon: maze_1d, for Q Learning, Q(state=1, action=0) === -2.0
2. 比较线性网络和卷积神经网络。因为信号的曲线可以看作是一种图形。
3. 多参数加权计算与单参数梯度函数的本质差异研究。
4. 学而不思则惘，思而不学则怠，RL 的 exploitation 和 exploration 考虑分开进行，而不是每一步骤进行？同时加大 exploration 的强度，跳出局部最优解。或者与其跳出最优解，不如上升格局，重新以更粗的粒度全局思考。我们的梯度本质上是去寻找一个曲面上不同部位的最低点，所以既要全局观察思考，也要细节勘测求值。
5. 一个神经网络不一定要解决一个问题的所有方面，它也可以只解决一个问题的部分方面，或者只解决某个维度？比如三维问题只解决其中一维？
6. Early-done algorithm for Generic AI.
