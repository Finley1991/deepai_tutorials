import gym
#创建环境：游戏名称
env = gym.make('MountainCar-v0')
# env = gym.make('CartPole-v1')
for i_episode in range(10):#循环多少个游戏片段
    #开始游戏，重置游戏初始状态
    observation = env.reset()
    for t in range(200):#每个游戏片段的游戏时间,MountainCar默认200,CartPole随机
        #显示游戏画面
        env.render()
        # print(observation)
        #随机采样：获得随机动作下的游戏画面样本
        action = env.action_space.sample()
        #获得每一帧画面输出的结果：St+1，Rt，是否结束，游戏信息
        observation_, reward, done, info = env.step(action)
        print(observation,action,reward,observation_,done,info)#st,at,rt,st+1,done,info
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()