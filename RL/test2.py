

import torch
from env import HCOENV
from stable_baselines3.common.env_checker import check_env
from gym.wrappers import TimeLimit
from stable_baselines3 import ppo
import time
import os
import matplotlib.pyplot as plt
import numpy as np

# torch.backends.mps.is_built()
LOAD = True


device = torch.device("mps")

env = HCOENV()

ENV = TimeLimit(HCOENV(),200)
total = 0

ppomodel = ppo.PPO.load("weights/ppo_3900k.zip", env = ENV, device=device)
for j in range(10):
    obs = ENV.reset()
    var = False
    mx = -100000
    eq = ""
    for jh in range(2000):
        action,_states= ppomodel.predict(obs)
        params = dict(zip(env.syms, ENV.weights))
        obs, rewards, dones, info = ENV.step(action)

        if rewards > mx:
            eq = ENV.print_eq()
            mx = rewards
        # if i % 20 == 0:
        #     print(f"Reward: {rewards}", end="\r")
        # plt.plot(list(range(i))[-100:], phi[-100:], "--b")
        # plt.ylim([-0.5,0.5])
        # plt.pause(0.05)
    total+=mx
    print(f"Trial Number: {j}")
    print(f"Max Reward = {100/mx}")
    print(eq)
print(f"Average reward: {100/(total/10)}")
    
    # print(f"Reward: {rewards}", end="\r")

    # print(rewards))