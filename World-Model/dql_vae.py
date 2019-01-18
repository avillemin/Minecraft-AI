import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch

from collections import namedtuple
import random

learning_rate = 0.0001
nb_outputs = 4
e = 0.2
nb_actions = 4 
Transition = namedtuple('Transition',
                        ('state','next_state','action','current_r','done'))
    
class brain(nn.Module):
    def __init__(self,latent_size,nb_actions):
        super(brain,self).__init__()
        self.fc1 = nn.Linear(in_features = latent_size, out_features = nb_actions)
    
    def forward(self,x):
        x = self.fc1(x)
        return x
    
class DQN():
    def __init__(self,gamma,latent_size,nb_actions):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        print("DQL running on GPU" if self.cuda else "DQL running on CPU")
        self.model=brain(latent_size,nb_actions).to(self.device)
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        self.nb_outputs = nb_outputs
        self.memory = []
        self.losses = []
        
        
    def select_action(self,state,epoch):
        #probs=F.softmax(self.model(torch.Tensor(state).detach())*temperature)
        #action = probs.multinomial(4)
        output = self.model(state).data
        action = np.argmax(output)
#        if np.random.rand(1)<e and (epoch < 600):
#            action = random.randint(0,self.nb_outputs-1)
#            print("Random action")
        return np.array(action)
    
    def learn(self, vae, batch_state, batch_next_state,batch_action,batch_reward,batch_not_done):
        self.optimizer.zero_grad()
        
        encoded_batch_state = vae.encode(batch_state).detach()
        encoded_batch_next_state = vae.encode(batch_next_state).detach()
        output = self.model(encoded_batch_state).gather(1, batch_action.to(self.device))
#        print('OUTPUT',output)
#        print('OUTPUT',output.size())
        next_output = self.model(encoded_batch_next_state).max(1)[0].detach()
        target = batch_reward.to(self.device) + torch.mul(next_output.unsqueeze(-1)*batch_not_done.to(self.device),self.gamma)
#        
#       
        td_loss = F.smooth_l1_loss(output, target)
#        print('td_loss :', td_loss)
        
        
        td_loss.backward()
        self.optimizer.step()
        self.losses.append(td_loss)
##        print(self.model(current_state)[0].detach())
        
    def replayMemoryPush(self,current_img, next_img, action, reward, done,memory_size):
        self.memory.append(Transition(current_img, next_img, torch.tensor(action,dtype=torch.long).view(1,1), torch.tensor(reward).view(1,1), torch.tensor(done,dtype = torch.float).view(1,1)))
        if len(self.memory) > memory_size:
            del self.memory[0]
    
    def replayMemorySample(self, batch_size):
        if len(self.memory)>batch_size:
            samples = random.sample(self.memory,batch_size)
        else:
            samples = random.sample(self.memory,len(self.memory))
        return Transition(*zip(*samples))
    
    