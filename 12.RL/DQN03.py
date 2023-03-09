import torch, gym, random
from torch import nn, optim

class QNet(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )


class Game:

    def __init__(self, exp_pool_size, explore=0.1):#由于环境中只有3个动作，探索值小一些收敛更快
        self.env = gym.make('MountainCar-v0')
        #定义经验池
        self.exp_pool = []
        #经验池尺寸上限
        self.exp_pool_size = exp_pool_size

        self.q_net = QNet()
        #探索值
        self.explore = explore

        self.loss_fn = nn.MSELoss()

        self.opt = optim.Adam(self.q_net.parameters())

    def __call__(self):
        is_render = False
        avg = 0
        while True:
            # 数据采样
            state = self.env.reset()
            # print(state.shape)
            #最终回报
            R = 0
            while True:
                if is_render:
                    #显示游戏画面
                    self.env.render()

                if len(self.exp_pool) >= self.exp_pool_size:
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

                next_state, reward, done, info = self.env.step(action)
                # print(next_state)
                #下一个状态的位置和速度
                position,velocity = next_state
                #将回报奖励改为各个方向的滑动绝对速度和绝对位置的乘积,奖励最大化就是速度最大化
                # reward = velocity
                reward = abs(velocity) * abs(position)
                R += reward
                self.exp_pool.append([state, reward, action, next_state, done])
                state = next_state
                if done:
                    avg = 0.95 * avg + 0.05 * R#计算总回报的期望
                    print(avg, R, self.env.spec.reward_threshold)
                    if avg > 0:
                        is_render = True
                    break


            # 训练
            #如果经验池装满，就开始训练
            if len(self.exp_pool) >= self.exp_pool_size:
                #从经验池中随机选择100条数据
                exps = random.choices(self.exp_pool, k=100)

                _state = torch.tensor([exp[0].tolist() for exp in exps])
                _reward = torch.tensor([[exp[1]] for exp in exps])
                _action = torch.tensor([[exp[2]] for exp in exps])
                _next_state = torch.tensor([exp[3].tolist() for exp in exps])
                _done = torch.tensor([[int(exp[4])] for exp in exps])

                # 估计值
                _Qs = self.q_net(_state)
                #根据动作索引获取Q值
                _Q = torch.gather(_Qs, 1, _action)

                # 目标值
                _next_Qs = self.q_net(_next_state)
                _max_Q = _next_Qs.max(dim=1, keepdim=True)[0]
                _target_Q = _reward + (1 - _done) * 0.9 * _max_Q#当前奖励加最大未来奖励（r（s）+max（s+1））

                loss = self.loss_fn(_Q, _target_Q.detach())
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()


if __name__ == '__main__':
    game = Game(10000)
    game()
