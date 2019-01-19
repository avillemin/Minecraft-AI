# Minecraft-AI

The goal of this project is to apply reseach papers on a Minecraft bot to resolve different tasks.

AI realized:
- Q-Learning : the bot has to reach a blue block without falling in the lava. The bot takes as input his current position. With this, he has to choose an action and wait for the reward. The maze stay the same during all the training and test.
- Deep Q-Learning : the bot has to reach a blue block in a map which changes at each try. He should not fall in the lava. The input is now the image seen by the bot. The AI is implemented with replay memory and eligibility trace.

I used pyTorch to realize the two bots.

The file DQL_bot.py create the Minecraft environment and build the map. Then, it will realize the actions chosen by the neural network created in DQL_network.py.

The next steps of this project will be to apply Augmented Random Search and A3C on the same environment to compare the results.

```bash
>> cd C:\Users\avillemin\Pictures\Malmo-0.36.0-Windows-64bit_withBoost_Python3.6\Minecraft
>> launchClient.bat
```

Environment used : https://github.com/Microsoft/malmo

N-step Q-learning : https://papoudakis.github.io/announcements/qlearning/  
Deep reinforcement learning https://arxiv.org/pdf/1312.5602.pdf  
Asynchronous Methods for Deep Reinforcement Learning : https://arxiv.org/pdf/1602.01783.pdf  

![Alt Text](https://github.com/avillemin/Minecraft-AI/blob/master/World-Model/VAE.png)
   
![Alt Text](https://github.com/avillemin/Minecraft-AI/blob/master/World-Model/VAE3.png)

# Reinforcement Learning

[Source] https://spinningup.openai.com/en/latest/index.html

## The Off-Policy Algorithms
   
Algorithms like DDPG and Q-Learning are off-policy, so they are able to reuse old data very efficiently. They gain this benefit by exploiting Bellman’s equations for optimality, which a Q-function can be trained to satisfy using any environment interaction data (as long as there’s enough experience from the high-reward areas in the environment).   
   
## The On-Policy Algorithms   
   
They don’t use old data, which makes them weaker on sample efficiency. But this is for a good reason: these algorithms directly optimize the objective you care about—policy performance—and it works out mathematically that you need on-policy data to calculate the updates. So, this family of algorithms trades off sample efficiency in favor of stability—but you can see the progression of techniques (from VPG to TRPO to PPO) working to make up the deficit on sample efficiency.   
   
      
## Kinds of RL Algorithms

![Alt text](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)
