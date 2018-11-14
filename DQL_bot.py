from __future__ import print_function

from builtins import range
import MalmoPython
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import DQL_network
import torch
import random

from skimage.transform import resize

actions = ['movenorth 1', 'movesouth 1', 'movewest 1', 'moveeast 1']
nb_actions = len(actions)
height = 30
width = 30
gamma = 0.3
ai = DQL_network.DQN_network(gamma,height,width,nb_actions)
max_retries = 3
nb_episodes = 1000
total_reward = []
victory = []
memory_size = 1000
batch_size = 100
last_action = 0

def img_process():
    video_frame = world_state.video_frames[-1].pixels
    video_frame = np.reshape(np.array(video_frame), (world_state.video_frames[-1].height,world_state.video_frames[-1].width,4))
    img = video_frame[:,25:-25,:3]
    img = resize(img, (height,width,3))
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32') / 255.
    img = torch.Tensor(img)
    img = img.view((1,3,height,width))
    return img
    
    
if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Hello world!</Summary>
              </About>
              
              <ServerSection>
                <ServerInitialConditions>
                <Time>
                <StartTime>1</StartTime>
                </Time>
                </ServerInitialConditions>
                <ServerHandlers>
                <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1"/>
                <DrawingDecorator>
                <DrawCuboid type="air" z2="13" y2="50" x2="9" z1="-2" y1="46" x1="-4"/>
                <DrawCuboid type="lava" z2="15" y2="45" x2="9" z1="-2" y1="45" x1="-4"/>
                <DrawCuboid type="sandstone" z2="17" y2="45" x2="6" z1="1" y1="45" x1="-1"/>
                <DrawBlock type="cobblestone" z="1" y="45" x="4"/>
                </DrawingDecorator>
                <ServerQuitFromTimeUp timeLimitMs="20000"/>
                <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
                </ServerSection>
              <AgentSection mode="Survival">
                <Name>MalmoTutorialBot</Name>
                <AgentStart>
                 <Placement z="1.5" y="46.0" x="4.5" yaw="0" pitch="45"/>
                 </AgentStart>
                <AgentHandlers>
                  <DiscreteMovementCommands autoFall="true"/>
                  <ObservationFromFullStats/>
                  <RewardForTouchingBlockType>
                    <Block type="lava" behaviour="onceOnly" reward="-1"/>                 
                    <Block type="lapis_block" behaviour="onceOnly" reward="1"/>                   
                    </RewardForTouchingBlockType>
                  <RewardForSendingCommand reward="-0.1"/>
                  <AgentQuitFromTouchingBlockType>
                    <Block type="lava"/>
                    <Block type="lapis_block"/>
                    </AgentQuitFromTouchingBlockType>
                  <VideoProducer viewpoint="0" want_depth="true">
                <Width>860</Width>
                <Height>480</Height>
            </VideoProducer>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

# Create default Malmo objects:

agent_host = MalmoPython.AgentHost()

try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

my_mission = MalmoPython.MissionSpec(missionXML, True)

for episode in range(nb_episodes):
    my_mission = MalmoPython.MissionSpec(missionXML, True)
    
    for x in range(-1,7):
        for z in range(2,14):
            if random.random()<0.15:
                my_mission.drawBlock( x,45,z,"lava")
                
    my_mission.drawBlock(np.random.randint(1,5),45,np.random.randint(3,14),"lapis_block")
    
    print()
    print('Repeat %d of %d' % ( episode, nb_episodes ))
    my_mission_record = MalmoPython.MissionRecordSpec()
    
    # Attempt to start a mission:
    
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)
    
    is_first_action = True
    cumulative_reward = 0
    # Loop until mission starts:
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
    
    # Loop until mission ends:
    while world_state.is_mission_running:
        is_first_action = True
        current_r = 0
        world_state = agent_host.getWorldState()
        if len(world_state.video_frames)>0:
            img = img_process()
            action = ai.select_action(img,episode)
            if cumulative_reward!=0:
                if (last_action == 0 and action == 1) or (last_action == 1 and action == 0) or (last_action == 2 and action == 3) or (last_action == 3 and action == 2):
                    action = np.random.randint(4)
            agent_host.sendCommand(actions[action])
            last_action = action

            while is_first_action:
                time.sleep(0.1)
                world_state = agent_host.getWorldState()
                for reward in world_state.rewards:
                    current_r += reward.getValue()
                    is_first_action = False
                if not world_state.is_mission_running:
                    break    
            
            print('Current reward :',current_r)
            
            if len(world_state.video_frames)>0 and current_r!=0:
                next_img = img_process()
                #ai.learn(img,next_img,current_r,action,abs(current_r)>0.5)
                ai.replayMemoryPush(img,next_img,action,current_r,abs(current_r)<0.5,memory_size)
            
                batch_img,batch_next_img,batch_action,batch_current_r,batch_not_done = ai.replayMemorySample(batch_size)
                batch_img = torch.cat(batch_img)
                batch_next_img = torch.cat(batch_next_img)
                batch_action = torch.cat(batch_action)
                batch_current_r = torch.cat(batch_current_r)
                batch_not_done = torch.cat(batch_not_done)
                ai.learn(batch_img,batch_next_img,batch_action,batch_current_r,batch_not_done)
            else:
                while True:
                    world_state = agent_host.getWorldState()
                    if len(world_state.video_frames)>0:
                        break
                print ('Next frame not available')
                
        cumulative_reward+=current_r
        for error in world_state.errors:
            print("Error:",error.text)
    total_reward.append(cumulative_reward)
    victory.append(1 if current_r>0.5 else 0)
    print()
    print("Mission ended")

l = []
for i in range(0,len(victory),100):
    l.append(np.mean(victory[i:i+99]))

plt.plot(l)
