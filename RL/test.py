

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

# ppomodel = ppo.PPO("CnnPolicy",ENV, verbose = 1, tensorboard_log="optim",device=device)
ppomodel = ppo.PPO.load("weights/ppo_2500k.zip", env = ENV, device=device)
total = 0
for j in range(5):
    obs = ENV.reset()
    var = False

    i = 1
    while not var:
        action,_states= ppomodel.predict(obs)
        params = dict(zip(env.syms, ENV.weights))

        obs, rewards, dones, info = ENV.step(action)

        if i % 20 == 0:
            print(f"Reward: {rewards}", end="\r")
        # plt.plot(list(range(i))[-100:], phi[-100:], "--b")
        # plt.ylim([-0.5,0.5])
        # plt.pause(0.05)
        var = rewards > 100000
        i+=1
    print("")
    print(f"Trial Number: {j}")
    print(ENV.return_matrices())
    total+=i
    # print(f"Reward: {rewards}", end="\r")

    # print(rewards))
print(f"Mean: {total/10}")
