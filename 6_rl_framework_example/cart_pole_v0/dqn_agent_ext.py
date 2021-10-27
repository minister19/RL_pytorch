from rl_m19.agent import DQNAgent


class DQNAgentExt(DQNAgent):
    def __init__(self, config):
        super().__init__(config)

    # q_eval, q_target: torch.tensor
    def gradient_descent(self, q_eval, q_target):
        loss = self.loss_fn(q_eval, q_target)  # compute loss
        self.optimizer.zero_grad()
        loss.backward()
        # 2020-08-13 Shawn: Sometimes, no clamp is better.
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()
