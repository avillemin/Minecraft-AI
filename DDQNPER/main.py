from Environment import Env, play
import matplotlib.pyplot as plt
from Agent import RandomAgent, DDQNPER_Agent
import numpy as np
import math

class HPS():
    def __init__(self):
        self.actions = ['movenorth 1', 'movesouth 1', 'movewest 1', 'moveeast 1']
        self.nb_actions = len(self.actions)
        self.height = 64
        self.width = 64
        self.img_channels = 3
        self.max_retries = 3
        self.nb_episodes_random = 400
        self.nb_episodes = 1000
        self.batch_size = 64
        self.total_reward = []
        self.previous_img = []
        self.mission_file = './maze.xml'
        self.memory_capacity = 100000
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.epsilon = 0.2
        self.huber_loss_delta = 2.0
        self.update_target_frequency = 15
        self.max_epsilon = 1
        self.min_epsilon = 0.1
        self.decreasing_rate = - math.log(0.01) / self.nb_episodes
        
hps = HPS()
plt.plot(hps.min_epsilon + (hps.max_epsilon - hps.min_epsilon) * np.exp(-hps.decreasing_rate * np.arange(hps.nb_episodes)))

env = Env(hps.mission_file)   
randomAgent = RandomAgent(hps)
play(env, hps, randomAgent, hps.nb_episodes_random) 

Agent = DDQNPER_Agent(hps)
Agent.memory = randomAgent.memory       
play(env, hps, Agent, hps.nb_episodes)
plt.plot(Agent.losses)

import torch
torch.save(Agent.optimizer.state_dict(),'models/DDQNPER_optimizer.pt')
torch.save(Agent.model.state_dict(),'models/DDQNPER_weights.pt')