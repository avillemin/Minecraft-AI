# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:30:28 2019

@author: avillemin
"""
import MalmoPython
import sys
import random
import numpy as np
import time
import torch

class Agent():
    def __init__(self, map_path):
        
#        if sys.version_info[0] == 2:
#            sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
#        else:
#            import functools
#            print = functools.partial(print, flush=True)
    
        self.agent_host = MalmoPython.AgentHost()
        self.agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
        self.agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)
        
        with open(map_path, 'r') as f:
            self.mission_xml = f.read()
        
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:',e)
            print(self.agent_host.getUsage())
            exit(1)
        if self.agent_host.receivedArgument("help"):
            print(self.agent_host.getUsage())
            exit(0)
            
    def init(self,max_retries):
        self.my_mission = MalmoPython.MissionSpec(self.mission_xml, True)
        for x in range(1,4):
            for z in range(2,14):
                if random.random()<0.15:
                    self.my_mission.drawBlock( x,45,z,"lava")
                    
        self.my_mission.drawBlock(np.random.randint(1,5),45,np.random.randint(3,14),"lapis_block")
        
        self.my_mission_record = MalmoPython.MissionRecordSpec()
    
        # Attempt to start a mission:
        
        for retry in range(max_retries):
            try:
                self.agent_host.startMission( self.my_mission,self.my_mission_record )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:",e)
                    exit(1)
                else:
                    time.sleep(2)
            
        return self.agent_host
    
def train_one_epoch(agent, hps, random_actions=True): 
    nb_action_done = 0
    batch_img = torch.empty([0,3,hps.height,hps.width])

    batch_rewards = []         # for measuring episode returns
        
    agent_host = agent.init(hps.max_retries)
    
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
        img = img_process(world_state, hps)
        batch_img = torch.cat((batch_img,img),0)
#        plt.axis('off')
#        plt.imshow(test)
#        plt.show()
        action = random_policy(hps, first_action = True)
        batch_action = [action]
        agent_host.sendCommand(hps.actions[action])
        nb_action_done+=1
    
    # Loop until mission ends:
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        current_r = 0
        while world_state.number_of_video_frames_since_last_state < 1 and world_state.is_mission_running:
            time.sleep(0.05)
            world_state = agent_host.getWorldState()  
            
        try:img = img_process(world_state, hps) 
        except:time.sleep(0.05)
        
        if world_state.number_of_rewards_since_last_state > 0:
            for reward in world_state.rewards:
                current_r += reward.getValue()
            cumulative_reward += current_r
            batch_rewards.append(current_r)
            batch_img = torch.cat((batch_img,img),0)
            action = random_policy(hps)
            batch_action.append(action)
            agent_host.sendCommand(hps.actions[action])
            nb_action_done+=1
    if current_r > 0.5:
        print('Success')
#    print('Number of moves: ',nb_action_done)
    return batch_img, batch_action[:-1], batch_rewards        

def img_process(world_state, hps):
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
    img = img.view((1,3,hps.height,hps.width))
    return img

def random_policy(hps, first_action = False):
    if first_action:
        acts = hps.actions + ['movewest 1']*3
        action = np.random.randint(len(acts))
        return action if action < hps.nb_actions else 2        
    else:
        acts = hps.actions + ['movesouth 1']*3
        action = np.random.randint(len(acts))
        return action if action < hps.nb_actions else 1