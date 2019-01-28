import matplotlib.pyplot as plt
import numpy as np
import torch
from Agent import Agent, train_one_epoch
from VAE import ConvVAE
from MDNRNN import MDNRNN 

class HPS():
    def __init__(self):
        self.actions = ['movenorth 1', 'movesouth 1', 'movewest 1', 'moveeast 1']
        self.nb_actions = len(self.actions)
        self.height = 64
        self.width = 64
        self.max_retries = 3
        self.latent_size = 16
        self.nb_episodes = 400
#        self.episode_length = 40
        self.total_reward = []
        self.previous_img = []
        self.action = 2
        self.mission_file = './maze.xml'
        self.batch_img = torch.empty([0,3,self.height,self.width])
        
hps = HPS()
agent = Agent(hps.mission_file)     

vae = ConvVAE(img_channels = 3, latent_size = hps.latent_size, learning_rate = 0.0001, load=True)
#vae.display_reconstruction(hps.batch_obs_temp,-1)

batch_obs, batch_act, batch_weight, batch_done = [], [], [], []
for episode in range(hps.nb_episodes):
    obs, acts, weights= train_one_epoch(agent,hps)
    if obs.size(0)>1:
        batch_obs.append(obs)
        batch_weight.append(torch.tensor(weights,dtype=torch.float32))
        batch_act.append(acts)
        batch_done.append(torch.tensor((len(acts)-1)*[0]+[1],dtype=torch.float32))
    if episode%int(hps.nb_episodes/10)==0:print('Epoch :',episode)

mdnrnn = MDNRNN(latent_size = hps.latent_size,input_size = hps.latent_size + len(hps.actions), rnn_hidden_size = 256)
input_obs, input_act, target_done, target_reward, target_obs, mask = mdnrnn.preprocess(vae, batch_obs, batch_act,batch_done, batch_weight, hps.nb_actions) 
mdnrnn.train(input_obs, target_obs, input_act, target_done, target_reward, mask, 3000)
plt.plot(mdnrnn.losses)

image = obs[0,:,:,:]
action = 1
encoded_image = vae.encode(image.unsqueeze(0))
(logpi, mu, sigma), hidden = mdnrnn.forward(encoded_image[0], action, hps)
rec_img = torch.normal(mu, sigma.exp())[0,0,:,:16]
vae.plot_encoded(rec_img)
vae.plot_encoded(image.unsqueeze(0),encoded=False)

img, done, reward = mdnrnn.play_in_dreams(image.unsqueeze(0), vae, hps)

vae.plot_encoded(img.unsqueeze(0))

vae.display_reconstruction(image.unsqueeze(0),-1)