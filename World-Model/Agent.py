# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:30:28 2019

@author: avillemin
"""
import MalmoPython
import sys
import random
import numpy as np
import os
import time

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