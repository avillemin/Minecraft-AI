import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch

import random

from collections import namedtuple
Transition = namedtuple('Transition',
                        ('state','next_state','action','current_r','done'))

class RandomAgent():
    def __init__(self, hps):
        self.actions = hps.actions
        self.nb_actions = hps.nb_actions
        self.memory = Memory(hps.memory_capacity)
        
    def select_action(self,state,epoch,first_action=False):
        if first_action:
            acts = self.actions + ['movewest 1']*3
            action = np.random.randint(len(acts))
            return action if action < self.nb_actions else 2        
        else:
            acts = self.actions + ['movesouth 1']*3
            action = np.random.randint(len(acts))
            return action if action < self.nb_actions else 1
        
    def observe(self,sample): # in (s, a, r, s_) format
        obs = sample[0]
        acts = sample[1]
        rewards = sample[2]
        next_obs = sample[3]
                    
        for step in range(obs.size(0)):
            self.memory.add(obs[step], next_obs[step], acts[step], rewards[step], abs(rewards[step])<0.5)   
                
    def replay(self):
        pass
    
class DQN_Agent():
    def __init__(self,hps):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        print("DDQNPER_Agent running on GPU" if self.cuda else "DDQNPER_Agent running on CPU")
        self.model = Brain(hps.height, hps.width, hps.nb_actions).to(self.device)
        self.gamma = hps.gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr = hps.learning_rate)
        self.nb_actions = hps.nb_actions
        self.memory = Memory(hps.memory_capacity)
        self.hps = hps
        self.steps = 0
        self.epsilon = hps.max_epsilon
        self.losses = []
        
    def predict(self, batch, target=False):
        return self.target_model(batch.to(self.device)) if target else self.model(batch.to(self.device))
        
    def select_action(self,state,epoch,first_action):
        Qvalues = self.model(torch.Tensor(state).to(self.device)).detach()
        action = np.argmax(Qvalues)
        if np.random.rand(1) < self.epsilon:
            action = random.randint(0,self.nb_actions-1)
        return np.array(action)
    
    def learn(self, batch_state, batch_next_state,batch_action,batch_reward,batch_not_done):
        output = self.model(batch_state.to(self.device)).gather(1, batch_action.to(self.device))
        next_output = self.model(batch_next_state.to(self.device)).max(1)[0].detach()
        target = batch_reward.to(self.device) + torch.mul(next_output.unsqueeze(-1)*batch_not_done.to(self.device),self.gamma)
        td_loss = self.huber_loss(output, target)
        
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
        self.losses+=[td_loss]              
    
    def replay(self):
        batch_img,batch_next_img,batch_action,batch_current_r,batch_not_done = self.memory.sample(self.hps.batch_size)   
        batch_img = torch.cat(batch_img)
        batch_next_img = torch.cat(batch_next_img)
        batch_action = torch.cat(batch_action)
        batch_current_r = torch.cat(batch_current_r)
        batch_not_done = torch.cat(batch_not_done)         
        self.learn(batch_img, batch_next_img,batch_action,batch_current_r,batch_not_done)
        
    def observe(self, sample):  # in (s, a, r, s_) format      sample = (batch_img[:-1], batch_action[:-1], batch_rewards, batch_img[1:])
        obs = sample[0]
        acts = sample[1]
        rewards = sample[2]
        next_obs = sample[3]
                    
        for step in range(obs.size(0)):
            self.memory.add(obs[step], next_obs[step], acts[step], rewards[step], abs(rewards[step])<0.5)       
            
        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = self.hps.min_epsilon + (self.hps.max_epsilon - self.hps.min_epsilon) * np.exp(-self.hps.decreasing_rate * self.steps)
            
    def huber_loss(self, y_true, y_pred):
        err = torch.tensor(y_true,dtype=torch.float32).to(self.device) - y_pred
    
        cond = abs(err) < self.hps.huber_loss_delta
        L2 = 0.5 * err**2
        L1 = self.hps.huber_loss_delta * (abs(err) - 0.5 * self.hps.huber_loss_delta)
    
        loss = torch.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(
    
        return loss.mean()
    
    def save(self,path=None):
        path = './models/DQN' if path==None else path
        torch.save(self.optimizer.state_dict(),path+'_optimizer.pt')
        torch.save(self.model.state_dict(),path+'_weights.pt')
        print('DDQNPER Model and Optimizer saved')
        
    def load(self,path=None):
        path = './models/DQN' if path==None else path
        self.model.load_state_dict(torch.load(path+'_weights.pt', map_location=self.device))
        self.optimizer.load_state_dict(torch.load(path+'_optimizer.pt', map_location=self.device))
        self.model.eval()
        print('DDQNPER Model and Optimizer loaded')
    
class Brain(nn.Module):
    def __init__(self,height,width,nb_outputs):
        super(Brain,self).__init__()
        self.convolution1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5)
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5)
        self.convolution3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.fc1 = nn.Linear(in_features = self.count_neurons((3, height, width)), out_features = 512)
        self.fc2 = nn.Linear(in_features = 512, out_features = nb_outputs)
              
    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x.data.view(1, -1).size(1)
    
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.contiguous().view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Memory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def add(self,current_img, next_img, action, reward, not_done):
        self.memory.append(Transition(current_img.unsqueeze(0), next_img.unsqueeze(0), torch.tensor(action,dtype=torch.long).view(1,1), torch.tensor(reward).view(1,1), torch.tensor(not_done,dtype = torch.float).view(1,1)))
        while len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        if len(self.memory)>batch_size:
            samples = random.sample(self.memory,batch_size)
        else:
            samples = random.sample(self.memory,len(self.memory))
        return Transition(*zip(*samples))  