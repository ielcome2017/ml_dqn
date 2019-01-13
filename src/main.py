import tensorflow as tf
import numpy as np
import gym

import random
from collections import deque
from gym.envs.classic_control.mountain_car import MountainCarEnv
from src.dqn import DQN

episdoe = 10000
step = 10000
env_name = "MountainCar-v0"
batch_size = 32
init_epsilon = 1.0
final_epsilon = 0.1
replay_size = 50000
train_start_size = 200
gamma = 0.9

# env = gym.make(env_name)
env = MountainCarEnv()

agent = DQN(init_epsilon, final_epsilon, env, replay_size, train_start_size, gamma)
for epd in range(episdoe):
    total_reward = 0
    state = env.reset()
    for s in range(step):
        env.render()
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        # print(reward)
        total_reward += reward
        agent.percieve(state, action, reward, next_state, done)
        if done:
            print("第%d次"%s)
            break
        state = next_state

    print("total reward is: ", total_reward)



