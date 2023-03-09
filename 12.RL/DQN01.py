import torch, gym, random
from torch import nn, optim


class QNet(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )


class Game:

    def __init__(self, exp_pool_size, explore=0.1):#由于环境中只有2个动作，探索值小一些收敛更快
        self.env = gym.make('CartPole-v1')

        self.exp_pool = []#当前样本库
        self.exp_pool_size = exp_pool_size#样本库阈值

        self.q_net = QNet()

        self.explore = explore#探索值

        self.loss_fn = nn.MSELoss()

        self.opt = optim.Adam(self.q_net.parameters())

    def __call__(self):
        is_render = False#默认关闭渲染
        avg = 0
        while True:#控制训练
            # 数据采样
            state = self.env.reset()#重置初始状态
            # print(state.shape)
            R = 0
            while True:#控制画面渲染
                if is_render: self.env.render()#是否开启渲染，默认关闭渲染
                # 显示游戏画面
                if len(self.exp_pool) >= self.exp_pool_size:#如果样本库数量已经达到最大数量，就更新样本库数据
                    self.exp_pool.pop(0)  # 把旧的经验删除掉
                    self.explore -= 1e-6  # 随着训练，agent对环境越来越熟悉，探索值可以逐渐减小
                    if random.random() > self.explore:#开发：找已知最大值
                        _state = torch.tensor(state).float()
                        # 将当前的状态输入到Q网络中，得到模型输出的Q值
                        Qs = self.q_net(_state[None, ...])
                        # 根据学习后的Q值得到更新后的动作
                        action = Qs.argmax(dim=1)[0].item()

                    else:#探索：随机获取一个动作
                        action = self.env.action_space.sample()

                else:  # 经验池没满就随机采样
                    action = self.env.action_space.sample()

                next_state, reward, done, info = self.env.step(action)#执行当前动作获得奖励和下一个状态
                R += reward#累计总奖励
                self.exp_pool.append([state, reward, action, next_state, done])#添加一条经验值
                state = next_state#迭代改变状态
                if done:
                    avg = 0.95 * avg + 0.05 * R#计算总回报的期望
                    print(avg, R, self.env.spec.reward_threshold)#总回报的期望(V价值)，总回报，阈值
                    if avg > 400:#当总回报的期望值大于阈值，开启渲染
                        is_render = True
                    break

            # 训练
            if len(self.exp_pool) >= self.exp_pool_size:#如果样本库数量不少于设置的阈值，就训练模型
                exps = random.choices(self.exp_pool, k=100)

                _state = torch.tensor([exp[0].tolist() for exp in exps])
                _reward = torch.tensor([[exp[1]] for exp in exps])
                _action = torch.tensor([[exp[2]] for exp in exps])
                _next_state = torch.tensor([exp[3].tolist() for exp in exps])
                _done = torch.tensor([[int(exp[4])] for exp in exps])

                # 估计值：当前状态下，执行某个动作后的Q值
                _Qs = self.q_net(_state)#Q网络，输出2个Q值
                _Q = torch.gather(_Qs, 1, _action)#根据当前动作索引获取Q值

                # 目标值：下一个状态下，奖励最大的Q值
                _next_Qs = self.q_net(_next_state)#目标网络
                _max_Q = _next_Qs.max(dim=1, keepdim=True)[0]
                _target_Q = _reward + (1 - _done) * 0.9 * _max_Q#当前奖励加最大未来奖励（r（s）+max（s+1））,折扣率为90%

                loss = self.loss_fn(_Q, _target_Q.detach())
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()


if __name__ == '__main__':
    game = Game(10000)
    game()
