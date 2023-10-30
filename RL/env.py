import sympy
from gym import spaces
import gym
import numpy as np
import math



def h(x):
  return -x * (x-0.25) * (x-1)

class HCOENV(gym.Env):
  metadata = {"render.modes": ["console"]}


  def __init__(self):
    super(HCOENV, self).__init__()
    self.action_space = spaces.Box(low=-1 * np.ones((8,)), high=np.ones((8,)), shape=(8,))
    self.observation_space = spaces.Box(low=-10 * np.ones((8,)), high=10 * np.ones((8,)), shape=(8,))

    self.weights = np.random.uniform(-1,1, (8,))
    self.syms = sympy.symbols("A:H")
    self.phi = sympy.Symbol("phi")
    self.eq = self.generate_eq()
    self.y = 0.25
    # self.eqs = sympy.solve(self.eq, self.phi)
  
  def ret_eq(self, params):
    roots = sympy.real_roots(self.eq.subs(params))
    all = []
    for i in roots:
      all.append(i.evalf())
    ind = np.abs(np.array(all) - 0.25).argmin()
    return all[ind]
  
  def return_matrices(self):
    gamma = np.zeros((2,2)).tolist()
    alpha = np.zeros_like(gamma).tolist()
    for ind,w in enumerate(self.weights[:4]):
      gamma[ind>1][ind%3 != 0] = w
    for ind,w in enumerate(self.weights[4:]):
      alpha[ind >1][ind%3 != 0] = w
    return alpha, gamma

  def from_matrices(self, alpha, gamma):
    out = []
    for ind in range(4):
      out.append(gamma[ind > 1][ind%3!=0])
    for ind in range(4):
      out.append(alpha[ind> 1][ind%3!=0])
    self.weights = np.array(out)

  def generate_eq(self):
    eq = 0
    # h = lambda n: -n*(n-0.25) * (n-1)

    for i,val in enumerate(self.syms):
      add = ((i % 2) == 1) * 0.5
      neg = (i > 3) * 2 - 1
      eq+=h(neg* self.phi + add) * val * -1 * neg
    return eq
  
  def reward(self, y, yhat):
    return (1 / (abs(y -yhat))) * 100

  def slope_rew(self,x):
    if x<0:
      return 1
    else:
      return -math.sqrt(float(x))


  def step(self,action):
    new_w = self.weights + action
    params = dict(zip(self.syms,new_w))
    filled = self.eq.subs(params)
    # sol = self.real(filled)
    reward = -100000.0
    done = False
    self.weights = new_w.astype(np.float32)
    sols = self.ret_eq(params)
    # try:
    #   sols = sympy.nsolve(filled, self.phi, (0.5))
    #   # print(f"alg: {val}, algo: {sols}")
    # except Exception as e:
    #   # print(f"alg: {val}, algo: NONE")

    #   return new_w, reward, done, {}
    slope = sympy.diff(filled).subs({self.phi: sols}).n()
    reward = self.reward(self.y,sols) * self.slope_rew(slope)
    done = abs(sols-0.25) < 1E-5
    return new_w.astype(np.float32), float(reward), bool(done), {}
  
  def print_eq(self):
    params = dict(zip(self.syms, self.weights))
    return self.eq.subs(params).subs({self.phi: sympy.Symbol("x")})
  
  def reset(self):
    self.weights = np.random.uniform(-1,1,(8,)).astype(np.float32)
    return self.weights
 