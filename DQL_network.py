import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch

from collections import namedtuple
import random

learning_rate = 0.02
nb_outputs = 4
e = 0.2
nb_actions = 4 
Transition = namedtuple('Transition',
                        ('state','next_state','action','current_r','done'))
    
class brain(nn.Module):
#    def __init__(self,height,width,nb_outputs):
#        super(brain,self).__init__()
#        self.convolution1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5)
#        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
#        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
#        self.fc1 = nn.Linear(in_features = self.count_neurons((3, height, width)), out_features = 40)
#        self.fc2 = nn.Linear(in_features = 40, out_features = nb_actions)
        
    def __init__(self,height,width,nb_outputs):
        super(brain,self).__init__()
#        self.convolution1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5)
#        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
#        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        self.fc1 = nn.Linear(in_features = 3*height*width, out_features = 200)
        self.fc2 = nn.Linear(in_features = 200, out_features = nb_actions)
        
    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x.data.view(1, -1).size(1)
    
    def forward(self,x):
#        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
#        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
#        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
#        print("Size: ",x.size())
        x = x.contiguous().view(x.size(0), -1)
#        print("Size: ",x.size())
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class DQN_network():
    def __init__(self,gamma,height,width,nb_outputs):
        self.model=brain(height,width,nb_outputs)
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        self.nb_outputs = nb_outputs
        self.memory = []
        
    def select_action(self,state,epoch):
        #probs=F.softmax(self.model(torch.Tensor(state).detach())*temperature)
        #action = probs.multinomial(4)
        output = self.model(torch.Tensor(state).detach()).data
#        plt.bar(['movenorth', 'movesouth', 'movewest', 'moveeast'],np.array(output)[0])
#        plt.show()
        action = np.argmax(output)
        if np.random.rand(1)<e and (epoch < 600):
            action = random.randint(0,self.nb_outputs-1)
            print("Random action")
        return np.array(action)
    
    def learn(self, batch_state, batch_next_state,batch_action,batch_reward,batch_not_done):
        output = self.model(batch_state).gather(1, batch_action)
        
#        print(output)
#        print('Reward :', reward)
#        print('Action :', action)
        next_output = self.model(batch_next_state).max(1)[0].detach()
#        print(next_output.shape)
#        print(batch_not_done.squeeze(1))
#        print(next_output)
        target = batch_reward + torch.mul(next_output.unsqueeze(-1)*batch_not_done,self.gamma)
        
       
        td_loss = F.smooth_l1_loss(output, target)
#        print('td_loss :', td_loss)
        
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
#        print(self.model(current_state)[0].detach())
        
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
    
    