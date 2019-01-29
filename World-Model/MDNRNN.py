# -*- coding: utf-8 -*-

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

import matplotlib.pyplot as plt

from torch.nn.utils.rnn import pad_sequence

class MDNRNN_network(nn.Module):
    def __init__(self, latent_size, input_size, rnn_hidden_size, nb_discrete_rewards = 3, dropout_rnn = 0, n_gaussians = 5, n_layers = 1):
        super(MDNRNN_network, self).__init__()
        self.nb_discrete_rewards = nb_discrete_rewards
        self.latent_size = latent_size
        self.rnn_hidden_size = rnn_hidden_size
        self.n_layers = n_layers
        self.n_gaussians = n_gaussians
        self.input_size = input_size
        self.rnn = nn.LSTM(input_size, rnn_hidden_size, n_layers, batch_first = True, dropout = dropout_rnn) 
        self.fc1 = nn.Linear(rnn_hidden_size, 3*n_gaussians*latent_size)
        self.fc_reward = nn.Linear(rnn_hidden_size, nb_discrete_rewards)
        self.fc_done = nn.Linear(rnn_hidden_size, 1)
        
    def forward(self, x, h):
        y, (h, c) = self.rnn(x, h)
        
        prob_rewards = F.softmax(self.fc_reward(y), dim=-1)
        prob_done = torch.sigmoid(self.fc_done(y))
        
        y = self.fc1(y)      
        pi, mu, logsigma = torch.split(y, self.n_gaussians*self.latent_size, dim=-1)
        pi = pi.view(-1, x.size(1), self.n_gaussians, self.latent_size)
        mu = mu.view(-1, x.size(1), self.n_gaussians, self.latent_size)
        logsigma = logsigma.view(-1, x.size(1), self.n_gaussians, self.latent_size)            
        logpi = F.log_softmax(pi, 2)
        
        return prob_rewards, prob_done, (logpi, mu, logsigma), (h, c)
        
    def init_hidden(self, batch_size):
        return torch.zeros(2, self.n_layers, batch_size, self.rnn_hidden_size) #number of layers, batch size, rnn_hidden_size
        
class MDNRNN():
    def __init__(self, latent_size, input_size, rnn_hidden_size, possible_rewards, dropout_rnn=0, n_gaussians = 5, n_layers = 1, learning_rate = 0.00001):
        self.cuda = torch.cuda.is_available()
        self.possible_rewards = possible_rewards
        self.device = torch.device("cuda" if self.cuda else "cpu")
        print("MDNRNN running on GPU" if self.cuda else "MDNRNN running on CPU")
        self.model = MDNRNN_network(latent_size, input_size, rnn_hidden_size,len(self.possible_rewards), dropout_rnn, n_gaussians, n_layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        self.losses = []
        self.losses_mdn = []
        self.losses_done = []
        self.losses_reward = []
        
    def train(self, input_obs, target_obs, input_actions, target_done, target_reward, mask, n_epochs):
        inputs = torch.cat((input_obs, input_actions),-1)
        past_loss = 0
        for epoch in range(n_epochs):
            hidden = self.model.init_hidden(inputs.size(0)).to(self.device)    
            self.optimizer.zero_grad()
            prob_rewards, prob_done, (logpi, mu, logsigma), hidden = self.model(inputs, hidden) 
            loss_mdn = self.loss_loglikelihood(logpi, mu, logsigma, target_obs, mask)
            loss_done, loss_reward = self.loss_objectives(prob_rewards, prob_done, mask, target_reward, target_done)
            loss = loss_mdn + loss_done + loss_reward
#            print(self.optimizer)
            if loss!=loss:
                print('WTF')
                print(past_loss)
                print(hidden)
                print(prob_rewards)
                break
            loss.backward()
            self.optimizer.step()
            self.losses.append(float(loss))
            self.losses_mdn.append(float(loss_mdn))
            self.losses_done.append(float(loss_done))
            self.losses_reward.append(float(loss_reward))
            past_loss = loss
        self.plot_losses()
            
    def loss_objectives(self, prob_rewards, prob_done, mask, target_reward, target_done):
        mse = nn.MSELoss(reduction='none')
        loss_done = mse(prob_done.squeeze(),target_done).sum()/mask.sum()
        loss_reward = (mse(prob_rewards,target_reward).mean(dim=-1)*mask).sum()/mask.sum()
        return loss_done, loss_reward
        
    def loss_loglikelihood(self, logpi, mu, logsigma, y, mask):
        y = y.unsqueeze(2)
        normal_dist = torch.distributions.Normal(loc=mu, scale=logsigma.exp())
        loss = logpi + normal_dist.log_prob(y)
        loss = torch.logsumexp(loss, 2)
        loss = torch.sum(loss, dim=-1)*mask
        loss = -loss.sum(dim = 1) / mask.sum(dim=1)
        return loss.mean()
    
    def preprocess(self, vae, list_obs, list_actions, list_dones, list_weights, hps):
        nb_actions = hps.nb_actions
        lenghts = [len(seq) for seq in list_actions]
        max_lenght = max(lenghts)
        mask = [[1]*i+[0]*(max_lenght-i) for i in lenghts]
        encoded_obs = [vae.encode(obs) for obs in list_obs]
        encoded_obs = pad_sequence(encoded_obs,batch_first=True)
        input_obs = encoded_obs[:,:-1,:]
        target_obs = encoded_obs[:,1:,:]
        one_hot = torch.eye(nb_actions)
        encoded_acts = [one_hot[act] for act in list_actions]
        input_act = pad_sequence(encoded_acts,batch_first=True) 
        target_done = pad_sequence(list_dones,batch_first=True)
        
        for i in range(len(list_weights)):
            for j in range(list_weights[i].size(0)):
                for k in range(len(hps.possible_rewards)):
                    if hps.possible_rewards[k] == list_weights[i][j]:
                        list_weights[i][j] = k
        target_reward = pad_sequence(list_weights,batch_first=True)
        target_reward = torch.tensor(target_reward,dtype=torch.long)
        target_reward = torch.eye(len(hps.possible_rewards))[target_reward]
        return input_obs.to(self.device), input_act.to(self.device), target_done.to(self.device), target_reward.to(self.device), target_obs.to(self.device), torch.tensor(mask, dtype=torch.float32).to(self.device)

                
    def forward(self, obs, action, hps, hidden = None, temperature = 1.0):
        obs = obs.unsqueeze(0).unsqueeze(0)
        action = torch.eye(hps.nb_actions)[action].unsqueeze(0).unsqueeze(0).to(self.device)
        inputs = torch.cat((obs, action),-1).to(self.device)
        if hidden == None:
            hidden = self.model.init_hidden(inputs.size(0)).to(self.device)    
        prob_rewards, prob_done, (logpi, mu, logsigma), hidden = self.model(inputs, hidden)
        return prob_rewards, prob_done, (logpi, mu, logsigma), hidden
    
    def play_in_dreams(self, first_frame, vae, hps):
        vae.plot_encoded(first_frame.to(self.device), encoded=False)
        current_frame = vae.encode(first_frame)[0] #Shape = [latent_size]
        hidden = None
        while True:
            action = int(input('Choose action: '))
            if action>=hps.nb_actions: break
            prob_rewards, prob_done, (logpi, mu, logsigma), hidden = self.forward(current_frame, action, hps, hidden)
            img = torch.normal(mu, logsigma.exp())[0,0,:,:]
            pi = logpi[0,0,:,:].exp()
            output = (pi*img).sum(dim = 0)
            vae.plot_encoded(output.unsqueeze(0))
            print('Reward: ',prob_rewards)
            print('Done: ',prob_done)
            current_frame = output
        return img, prob_done, prob_rewards
    
    def plot_losses(self):
        fig = plt.figure(figsize = (12,12))
        plt.subplot(2,2,1)
        plt.plot(self.losses)
        plt.title('Global loss', size=20)
        plt.subplot(2,2,2)
        plt.plot(self.losses_mdn,'r')
        plt.title('MDN loss', size=20)
        plt.subplot(2,2,3)
        plt.plot(self.losses_done, 'm')
        plt.title('Done loss', size=20)
        plt.subplot(2,2,4)
        plt.plot(self.losses_reward, 'g')
        plt.title('Reward loss', size=20)
        fig.show()
        
