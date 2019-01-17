from __future__ import print_function

from builtins import range
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from Agent import Agent

actions = ['movenorth 1', 'movesouth 1', 'movewest 1', 'moveeast 1', 'movewest 1', 'movewest 1'] #increase the probability to go straight
nb_actions = len(actions)
height = 64
width = 64
max_retries = 3
nb_episodes = 500
total_reward = []
last_frame = []
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

for episode in range(nb_episodes):   
    
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
    
    # Loop until mission ends:
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        current_r = 0
        while world_state.number_of_video_frames_since_last_state < 1 and world_state.is_mission_running:
            time.sleep(0.05)
            world_state = agent_host.getWorldState()  
            
        try:img = img_process() 
        except:time.sleep(0.05)
        
        if world_state.number_of_rewards_since_last_state > 0:
            for reward in world_state.rewards:
                current_r += reward.getValue()
            cumulative_reward += current_r
#            print(current_r)
#            plt.axis('off')
#            plt.imshow(test)
#            plt.show()
            batch_img = torch.cat((batch_img,img),0)
            action = np.random.randint(nb_actions)
            agent_host.sendCommand(actions[action])
            
        
            
print('Number of images: ',batch_img.size())

from VAE import ConvVAE
vae = ConvVAE(img_channels = 3, latent_size = 32, learning_rate = 0.0001)
temp = vae.train(batch_img,batch_size=64,nb_epochs = 1)

vae.display_reconstruction(batch_img,31)

#vae.save()  
#vae.load()