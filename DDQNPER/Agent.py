import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch
import math

import random

from SumTree import SumTree

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
        
        error = abs(np.array(rewards))        
        for step in range(obs.size(0)): 
            self.memory.add(error[step],(obs[step], acts[step], rewards[step], next_obs[step]))   
                
    def replay(self):
        pass
    
class DDQNPER_Agent():
    def __init__(self,hps):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        print("DDQNPER_Agent running on GPU" if self.cuda else "DDQNPER_Agent running on CPU")
        self.model = Brain(hps.height, hps.width, hps.nb_actions).to(self.device)
        self.target_model = Brain(hps.height, hps.width, hps.nb_actions).to(self.device)
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
        Qvalues = self.predict(torch.Tensor(state)).detach()
#        print(Qvalues)
        action = np.argmax(Qvalues)
        if np.random.rand(1) < self.epsilon:
            action = random.randint(0,self.nb_actions-1)
#            print("Random action")
        return np.array(action)
    
    def learn(self, x, y_true):
        y_pred = self.model(torch.tensor(x,dtype=torch.float32).to(self.device)) 
#        l1_loss = torch.nn.SmoothL1Loss()
#        td_loss = l1_loss(y_true, y_pred)
        td_loss = self.huber_loss(y_true, y_pred)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
        
        self.losses+=[td_loss]
            
    def updateTargetModel(self):
        self.target_model.load_state_dict(self.model.state_dict())    
    
    def replay(self):
        batch = self.memory.sample(self.hps.batch_size)
        x, y, errors = self._getTargets(batch)
        
        for i in range(len(batch)): #update errors
            idx = batch[i][0]
            self.memory.update(idx, errors[i])
            
        self.learn(x, y)
        
    def observe(self, sample):  # in (s, a, r, s_) format      sample = (batch_img[:-1], batch_action[:-1], batch_rewards, batch_img[1:])
        obs = sample[0]
        acts = sample[1]
        rewards = sample[2]
        next_obs = sample[3]
        
        if obs.size(0)>1:
            error = abs(np.array(rewards))     
            
        for step in range(obs.size(0)): 
            x, y, error = self._getTargets([(0, [obs[step], acts[step], rewards[step], next_obs[step]])])
            self.memory.add(error[0],(obs[step], acts[step], rewards[step], next_obs[step]))   
        
        if self.steps % self.hps.update_target_frequency == 0:
            self.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = self.hps.min_epsilon + (self.hps.max_epsilon - self.hps.min_epsilon) * np.exp(-self.hps.decreasing_rate * self.steps)
        
    def _getTargets(self, batch):
        batch_obs = torch.cat([obs[1][0].unsqueeze(0) for obs in batch])
        batch_next_obs = torch.cat([obs[1][3].unsqueeze(0) for obs in batch])

        Qvalues_pred = self.predict(batch_obs).detach()
        next_Qvalues_pred = self.predict(batch_next_obs).detach()
        next_Qvalues_target = self.predict(batch_next_obs, target=True).detach()

        x = np.zeros((len(batch), self.hps.img_channels, self.hps.width, self.hps.height))
        y = np.zeros((len(batch), self.hps.nb_actions))
        errors = np.zeros(len(batch))
        
        for i in range(len(batch)):
            rollout = batch[i][1]
            obs = rollout[0]
            action = rollout[1]
            reward = rollout[2] #; next_obs = rollout[3]
            t = Qvalues_pred[i]
            oldVal = t[action]
            t[action] = reward + self.hps.gamma*next_Qvalues_target[i][np.argmax(next_Qvalues_pred[i])]*(abs(reward)<0.5)  # double DQN

            x[i] = obs
            y[i] = t
            errors[i] = abs(oldVal - t[action])
            
        return (x, y, errors)
    
    def huber_loss(self, y_true, y_pred):
        err = torch.tensor(y_true,dtype=torch.float32).to(self.device) - y_pred
    
        cond = abs(err) < self.hps.huber_loss_delta
        L2 = 0.5 * err**2
        L1 = self.hps.huber_loss_delta * (abs(err) - 0.5 * self.hps.huber_loss_delta)
    
        loss = torch.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(
    
        return loss.mean()
    
    def save(self,path=None):
        path = './models/DDQNPER' if path==None else path
        torch.save(self.optimizer.state_dict(),path+'_optimizer.pt')
        torch.save(self.model.state_dict(),path+'_weights.pt')
        print('DDQNPER Model and Optimizer saved')
        
    def load(self,path=None):
        path = './models/DDQNPER' if path==None else path
        self.model.load_state_dict(torch.load(path+'_weights.pt', map_location=self.device))
        self.target_model.load_state_dict(torch.load(path+'_weights.pt', map_location=self.device))
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
    
class Memory():   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )
#            batch.append(data)
        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)    