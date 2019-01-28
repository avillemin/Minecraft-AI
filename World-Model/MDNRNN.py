# -*- coding: utf-8 -*-

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from torch.nn.utils.rnn import pad_sequence

class MDNRNN_network(nn.Module):
    def __init__(self, latent_size, input_size, rnn_hidden_size, dropout_rnn=0, n_gaussians = 5, n_layers = 1):
        super(MDNRNN_network, self).__init__()
        self.latent_size = latent_size
        self.rnn_hidden_size = rnn_hidden_size
        self.n_layers = n_layers
        self.n_gaussians = n_gaussians
        self.input_size = input_size
        self.rnn = nn.LSTM(input_size, rnn_hidden_size, n_layers, batch_first = True, dropout = dropout_rnn) 
        self.fc1 = nn.Linear(rnn_hidden_size, 3*n_gaussians*latent_size)
        
    def forward(self, x, h):
        y, (h, c) = self.rnn(x, h)
        y = self.fc1(y)      
        pi, mu, logsigma = torch.split(y, self.n_gaussians*self.latent_size, dim=-1)
        pi = pi.view(-1, x.size(1), self.n_gaussians, self.latent_size)
        mu = mu.view(-1, x.size(1), self.n_gaussians, self.latent_size)
        logsigma = logsigma.view(-1, x.size(1), self.n_gaussians, self.latent_size)            
        logpi = F.log_softmax(pi, 2)
        
        return (logpi, mu, logsigma), (h, c)
        
    def init_hidden(self, batch_size):
        return torch.zeros(2, self.n_layers, batch_size, self.rnn_hidden_size) #number of layers, batch size, rnn_hidden_size
        
class MDNRNN():
    def __init__(self, latent_size, input_size, rnn_hidden_size, dropout_rnn=0, n_gaussians = 5, n_layers = 1, learning_rate = 0.0001):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        print("MDNRNN running on GPU" if self.cuda else "MDNRNN running on CPU")
        self.model = MDNRNN_network(latent_size + 2, input_size, rnn_hidden_size, dropout_rnn, n_gaussians, n_layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        self.losses = []
        
    def train(self, input_obs, target_obs, input_actions, target_done, target_reward, mask, n_epochs):
        inputs = torch.cat((input_obs, input_actions),-1)
        outputs = torch.cat((target_obs, target_done.unsqueeze(-1), target_reward.unsqueeze(-1)),-1)
        for epoch in range(n_epochs):
            hidden = self.model.init_hidden(inputs.size(0)).to(self.device)    

            (logpi, mu, logsigma), hidden = self.model(inputs, hidden)
            
            self.optimizer.zero_grad()
            loss = self.loss_loglikelihood(logpi, mu, logsigma, outputs, mask)
            loss.backward()
            self.optimizer.step()
            self.losses.append(float(loss))
            
    def forward(self, obs, action, hps, hidden = None, temperature = 1.0):
        obs = obs.unsqueeze(0).unsqueeze(0)
        action = torch.eye(hps.nb_actions)[action].unsqueeze(0).unsqueeze(0).to(self.device)
        inputs = torch.cat((obs, action),-1).to(self.device)
        if hidden == None:
            hidden = self.model.init_hidden(inputs.size(0)).to(self.device)    
        (logpi, mu, logsigma), hidden = self.model(inputs, hidden)
        return (logpi, mu, logsigma), hidden
        
    def loss_loglikelihood(self, logpi, mu, logsigma, y, mask):
        y = y.unsqueeze(2)
        normal_dist = torch.distributions.Normal(loc=mu, scale=logsigma.exp())
        loss = logpi + normal_dist.log_prob(y)
        loss = torch.logsumexp(loss, 2)
        loss = torch.sum(loss, dim=-1)*mask
        loss = -loss.sum(dim = 1) / mask.sum(dim=1)
        return loss.mean()
    
    def preprocess(self, vae, list_obs, list_actions, list_dones, list_weights, nb_actions):
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
        target_reward = pad_sequence(list_weights,batch_first=True)
        return input_obs.to(self.device), input_act.to(self.device), target_done.to(self.device), target_reward.to(self.device), target_obs.to(self.device), torch.tensor(mask, dtype=torch.float32).to(self.device)
    
    def play_in_dreams(self, first_frame, vae, hps):
        vae.plot_encoded(first_frame.to(self.device), encoded=False)
        current_frame = vae.encode(first_frame)[0] #Shape = [latent_size]
        print(current_frame.size())
        hidden = None
        while True:
            action = int(input('Choose action: '))
            if action>hps.nb_actions: break
            (logpi, mu, logsigma), hidden = self.forward(current_frame, action, hps, hidden)
            print(mu)
            img = torch.normal(mu, logsigma.exp())[0,0,:,:]
            pi = logpi[0,0,:,:].exp()
            output = (pi*img).sum(dim = 0)
            img = output[:16]
            done = output[16]
            reward = output[17]
            vae.plot_encoded(img.unsqueeze(0))
            print('Reward: ',reward)
            print('Done: ',done)
            current_frame = img
        return img, done, reward
        
