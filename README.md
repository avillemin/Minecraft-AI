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
   
# Kinds of RL Algorithms

![Alt text](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)

## Model-Free vs Model-Based RL

   One of the most important branching points in an RL algorithm is the question of whether the agent has access to (or learns) a model of the environment. By a model of the environment, we mean a function which predicts state transitions and rewards.   
   The main upside to having a model is that it allows the agent to plan by thinking ahead, seeing what would happen for a range of possible choices, and explicitly deciding between its options.    
   Algorithms which use a model are called model-based methods, and those that don’t are called model-free. While model-free methods forego the potential gains in sample efficiency from using a model, they tend to be easier to implement and tune. As of the time of writing this introduction (September 2018), model-free methods are more popular and have been more extensively developed and tested than model-based methods.   

What to learn:
- policies, either stochastic or deterministic,
- action-value functions (Q-functions),
- value functions,
- and/or environment models.

## Simplest Policy Gradient   
   
We have our policy π that has a parameter θ. This π outputs a probability distribution of actions.
   
<p align="center"><img src="https://cdn-images-1.medium.com/max/1000/0*354cfoILK19WFTWa."></p>

[Source] https://medium.freecodecamp.org/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f
Q-Learning: value-based reinforcement learning algorithms. To choose which action to take given a state, we take the action with the highest Q-value (maximum expected future reward I will get at each state). As a consequence, in **value-based learning**, a policy exists only because of these action-value estimates.   
In **policy-based** methods, instead of learning a value function that tells us what is the expected sum of rewards given a state and an action, we learn directly the policy function that maps state to action (select actions without using a value function). It means that we directly try to optimize our policy function π without worrying about a value function. We’ll directly parameterize π (select an action without a value function).

A policy can be either deterministic or stochastic.   
A deterministic policy is policy that maps state to actions. You give it a state and the function returns an action to take. Deterministic policies are used in deterministic environments. These are environments where the actions taken determine the outcome. There is no uncertainty. For instance, when you play chess and you move your pawn from A2 to A3, you’re sure that your pawn will move to A3.
On the other hand, a stochastic policy outputs a probability distribution over actions. It means that instead of being sure of taking action a (for instance left), there is a probability we’ll take a different one (in this case 30% that we take south).   
   
![Alt Text](https://cdn-images-1.medium.com/max/1000/1*YCABimP7x1wZZZKqz2CoyQ.png)   
   
The stochastic policy is used when the environment is uncertain. We call this process a Partially Observable Markov Decision Process (POMDP).   
Most of the time we’ll use this second type of policy.    

**Advantages**: But Deep Q Learning is really great! Why using policy-based reinforcement learning methods?   
- For one, policy-based methods have better convergence properties. The problem with value-based methods is that they can have a big oscillation while training. This is because the choice of action may change dramatically for an arbitrarily small change in the estimated action values. On the other hand, with policy gradient, we just follow the gradient to find the best parameters. We see a smooth update of our policy at each step.
- Policy gradients are more effective in high dimensional action spaces
![Alt Text](https://cdn-images-1.medium.com/max/1000/1*_hAkM4RIxjKjKqAYFR_9CQ.png)
   
- A third advantage is that policy gradient can learn a stochastic policy, while value functions can’t.   
A stochastic policy allows our agent to explore the state space without always taking the same action. This is because it outputs a probability distribution over actions. As a consequence, it handles the exploration/exploitation trade off without hard coding it

**Disadvantages**:   
Naturally, Policy gradients have one big disadvantage. A lot of the time, they converge on a local maximum rather than on the global optimum.   
Instead of Deep Q-Learning, which always tries to reach the maximum, policy gradients converge slower, step by step. They can take longer to train.   

Here, we consider the case of a stochastic, parameterized policy, ![](https://spinningup.openai.com/en/latest/_images/math/80088cfe6126980142c5447a9cb12f69ee7fa333.svg). We aim to maximize the expected return ![Alt Text](https://spinningup.openai.com/en/latest/_images/math/48ffbf0dd0274a46574e145ea23e4c174f6dfaa3.svg). For the purposes of this derivation, we’ll take R(tau) to give the finite-horizon undiscounted return, but the derivation for the infinite-horizon discounted return setting is almost identical.
   
There are two steps:   
- Measure the quality of a π (policy) with a policy score function J(θ)   
- Use policy gradient ascent to find the best parameter θ that improves our π.   
   
We would like to optimize the policy by gradient ascent, eg
<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/237c86938ce2e9de91040e4090f79c6a1125fc00.svg"></p> 
This is an expectation, which means that we can estimate it with a sample mean. If we collect a set of trajectories where each trajectory is obtained by letting the agent act in the environment using the policy, the policy gradient can be estimated with
<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/a8ec906d99c7cb540ef0df80d86fa1bca0f33a79.svg"></p> 
