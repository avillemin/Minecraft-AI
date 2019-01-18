from __future__ import print_function

from builtins import range
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from Agent import Agent

actions = ['movenorth 1', 'movesouth 1', 'movewest 1', 'moveeast 1', 'movesouth 1', 'movesouth 1', 'movesouth 1'] #increase the probability to go straight
nb_actions = 4
height = 64
width = 64
max_retries = 3
latent_size = 16
nb_episodes = 100
total_reward = []
previous_img = []
action = 2
mission_file = './maze.xml'
batch_img = torch.empty([0,3,height,width])

def img_process():
    video_frame = world_state.video_frames[-1].pixels
    video_frame = np.reshape(np.array(video_frame), (world_state.video_frames[-1].height,world_state.video_frames[-1].width,4))
    global test
    test = video_frame[:,:,:3]
    img = video_frame[:,:,:3]
#    plt.imshow(img)
#    plt.show()
#    img = resize(img, (height,width,3))
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32') / 255.
    img = torch.Tensor(img)
    img = img.view((1,3,height,width))
    return img

agent = Agent(mission_file)

from dql_vae import DQN
dql = DQN(gamma=0.8,latent_size=latent_size,nb_actions=nb_actions)

for episode in range(nb_episodes):   
    nb_action_done = 0
    print()
    print('Repeat %d of %d' % ( episode, nb_episodes ))
    
    agent_host = agent.init(max_retries)
    
    cumulative_reward = 0
    # Loop until mission starts:
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
    
    #First action
    if world_state.is_mission_running:
        while len(world_state.video_frames)<1 and world_state.is_mission_running:
            time.sleep(0.05)
            world_state = agent_host.getWorldState() 
        img = img_process()
        batch_img = torch.cat((batch_img,img),0)
#        plt.axis('off')
#        plt.imshow(test)
#        plt.show()
        agent_host.sendCommand(actions[2])
        nb_action_done+=1
    
    # Loop until mission ends:
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        current_r = 0
        while world_state.number_of_video_frames_since_last_state < 1 and world_state.is_mission_running:
            time.sleep(0.05)
            world_state = agent_host.getWorldState()  
            
        try:previous_img,img = img,img_process() 
        except:time.sleep(0.05)
        
        if world_state.number_of_rewards_since_last_state > 0:
            for reward in world_state.rewards:
                current_r += reward.getValue()
            cumulative_reward += current_r
            dql.replayMemoryPush(previous_img, img, action, current_r, abs(current_r)<0.5,1000)
            batch_img = torch.cat((batch_img,img),0)
            action = np.random.randint(nb_actions)
            agent_host.sendCommand(actions[action])
            nb_action_done+=1
    if current_r > 0.5:
        print('Success')
    print('Number of moves: ',nb_action_done)
         
        
            
#print('Number of images: ',batch_img.size())
#
from VAE import ConvVAE
vae = ConvVAE(img_channels = 3, latent_size = latent_size, learning_rate = 0.0001)
temp = vae.train(batch_img,batch_size=128,nb_epochs = 50)

vae.display_reconstruction(batch_img,100)
#vae.save() 
#vae.load('./models/test')
    
# LEARNING OF THE DQL NETWORK
for i in range(3000):
    batch_imgs,batch_next_imgs,batch_actions,batch_current_rs,batch_not_dones = dql.replayMemorySample(128*2*2)    
    batch_imgs = torch.cat(batch_imgs)
    batch_next_imgs = torch.cat(batch_next_imgs)
    batch_actions = torch.cat(batch_actions)
    batch_current_rs = torch.cat(batch_current_rs)
    batch_not_dones = torch.cat(batch_not_dones) 
    dql.learn(vae,batch_imgs,batch_next_imgs,batch_actions,batch_current_rs,batch_not_dones) 
        
#events = dql.memory
#dql.memory = events

for episode in range(nb_episodes):   
    nb_action_done = 0
    print()
    print('Repeat %d of %d' % ( episode, nb_episodes ))
    
    agent_host = agent.init(max_retries)
    
    cumulative_reward = 0
    # Loop until mission starts:
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
    
    #First action
    if world_state.is_mission_running:
        while len(world_state.video_frames)<1 and world_state.is_mission_running:
            time.sleep(0.05)
            world_state = agent_host.getWorldState() 
        img = img_process()
        batch_img = torch.cat((batch_img,img),0)
#        plt.axis('off')
#        plt.imshow(test)
#        plt.show()
        agent_host.sendCommand(actions[2])
        nb_action_done+=1
    
    # Loop until mission ends:
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        current_r = 0
        while world_state.number_of_video_frames_since_last_state < 1 and world_state.is_mission_running:
            time.sleep(0.05)
            world_state = agent_host.getWorldState()  
            
        try:previous_img,img = img,img_process() 
        except:time.sleep(0.05)
        
        if world_state.number_of_rewards_since_last_state > 0:
            for reward in world_state.rewards:
                current_r += reward.getValue()
            cumulative_reward += current_r
            dql.replayMemoryPush(previous_img, img, action, current_r, abs(current_r)<0.5,1000)
            batch_img = torch.cat((batch_img,img),0)
            action = dql.select_action(vae.encode(img),episode)
            agent_host.sendCommand(actions[action])
            nb_action_done+=1
    if current_r > 0.5:
        print('Success')
    print('Number of moves: ',nb_action_done)