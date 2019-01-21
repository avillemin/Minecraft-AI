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

<p align="center"><img src="https://github.com/avillemin/Minecraft-AI/blob/master/World-Model/VAE.png"></p>
   
# Reinforcement Learning

[Source] https://spinningup.openai.com/en/latest/index.html

## The Off-Policy Algorithms
   
Algorithms like DDPG and Q-Learning are off-policy, so they are able to reuse old data very efficiently. They gain this benefit by exploiting Bellman’s equations for optimality, which a Q-function can be trained to satisfy using any environment interaction data (as long as there’s enough experience from the high-reward areas in the environment).   
   
## The On-Policy Algorithms   
   
They don’t use old data, which makes them weaker on sample efficiency. But this is for a good reason: these algorithms directly optimize the objective you care about—policy performance—and it works out mathematically that you need on-policy data to calculate the updates. So, this family of algorithms trades off sample efficiency in favor of stability—but you can see the progression of techniques (from VPG to TRPO to PPO) working to make up the deficit on sample efficiency.   
   
### Bellman Equations   
Finite-horizon undiscounted return:  
<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/ce20ca1d911ea7b3b9161000c52ed750ec75cc14.svg"></p>  
    
Infinite-horizon discounted return:   
<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/196d139b5647c5a777ebe6ddfce278f8b0736156.svg"></p> 
       
<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/d68c87fb46a17130d25b040ad23fb2409ae764a1.svg"></p>  
   
The optimal policy in s will select whichever action maximizes the expected return from starting in s. As a result, if we have Q*, we can directly obtain the optimal action, a*(s), via 
<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/4a5e5d2aad03e229c014d1990a78732de2144b9a.svg"></p>

There is a key connection between the value function and the action-value function:   
<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/45dfe72ba680d985e158c2fef08ddfbb9c5f57a6.svg"></p>
Advantage function:   
<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/a596eb68ba26e424afaff142ae747d5cffd2be60.svg"></p>
   
## Kinds of RL Algorithms

![Alt text](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)
