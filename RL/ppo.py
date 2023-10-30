import torch
from env import HCOENV
from stable_baselines3.common.env_checker import check_env
from gym.wrappers import TimeLimit
from stable_baselines3 import ppo
import os
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("-d","--device",help="Y or N for if using apple silicon")
parser.add_argument("-l","--load",help="Y or N for if using previous weight checkpoint")
parser.add_argument("-i","--iterations", help="number of training iterations")
args = parser.parse_args()
# torch.backends.mps.is_built()
LOAD = True
if args.load == "n" or args.load == "N":
    LOAD = False


device = torch.device("mps")
ITERATIONS = 200000
if args.iterations is not None:
    ITERATIONS = int(args.iterations)

# print(type(HCOENV())
# check_env(HCOENV())
ENV = TimeLimit(HCOENV(),500)

ppomodel = ppo.PPO("MlpPolicy",ENV, verbose = 1, tensorboard_log="optim",device=device)
dct = {"k":10**3}
max = 0
if LOAD:
    weights = os.listdir("weights")
    filename = ""
    for w in weights:
        splt = w.split(".")[0].split("_")[1]
        num = dct[splt[-1]] * int(splt[:-1])
        if num > max:
            filename = w
            max = num
    print(f"loading: {filename}")
    ppomodel = ppo.PPO.load(f"weights/{filename}", env = ENV, device=device)

ppomodel.learn(ITERATIONS)
ppomodel.save(f"weights/ppo_{int(int(max)/1000+ITERATIONS/1000)}k.zip")
