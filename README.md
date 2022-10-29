# Intro

This repo records Shuang Gao's AI learning progress.

# \_env Setup

```
pip install -U autopep8
pip install numpy
pip install gym
pip3 install torch torchvision torchaudio
pip install matplotlib
pip install pyglet
pip install -e .\5_rl_framework\.
```

# Diaries

2021-04-29 Shawn: ask for leave. OT in ABB, cannot get home.
2021-05-02 Shawn: ask for leave. Travelling to Tangyin.
2021-05-03 Shawn: ask for leave. Travelling to Tangyin.
2021-05-13 Shawn: ask for leave. Play SC2.
2021-05-13 Shawn: ask for leave. Xiaoyun visits.
2021-09-22 Shawn: ask for leave. Illness recovery.
2021-09-24 Shawn: ask for leave. Play Dota2.
2021-09-25 Shawn: ask for leave. Take a rest.
2021-09-30 Shawn: ask for leave. Play SC2.
2021-10-01 Shawn: ask for leave. Play Dyson Sphere Program.
2021-10-02 Shawn: ask for leave. Attend Leo's wedding.

# TODO

1. Research that tuning reward such that q table reaches convergence.
   phenomenon: maze_1d, for Q Learning, Q(state=1, action=0) === -2.0
2. 研究神经元符号翻转，自动使输入的正负号翻转，实现收敛。
3. 比较线性网络和卷积神经网络。因为信号的曲线可以看作是一种图形。
4. 多参数加权计算与单参数梯度函数的本质差异研究。
5. 研究模型输出维度与网络维度的关系。
6. 学而不思则惘，思而不学则怠，RL 的 exploitation 和 exploration 考虑分开进行，而不是每一步骤进行？同时加大 exploration 的强度，跳出局部最优解。或者与其跳出最优解，不如上升格局，重新以更粗的粒度全局思考。我们的梯度本质上是去寻找一个曲面上不同部位的最低点，所以既要全局观察思考，也要细节勘测求值。
7. 一个神经网络不一定要解决一个问题的所有方面，它也可以只解决一个问题的部分方面，或者只解决某个维度？比如三维问题只解决其中一维？
8. action_transits_quota algorithm for Generic AI.
9. Find the relationships:
   eps decay - memory capacity
   batch size - memory capacity
   eps decay - data trained
   batch size - data trained
10. 研究是否可以用二进制替代 action space，这样最后一层网络尺寸缩小为 log(a_dim).比如 00，01，10，11 分别代表四种 action。
11. 设计比较器，可以分析 input 与所需 output 的相关性（协方差），进而剥夺 input，这样网络仍然是全连接，不需要 dropout。
12. 研究 multiple activation functions，一个 Relu，一个 Step（输出 bool），一个导数，一个幂指数
