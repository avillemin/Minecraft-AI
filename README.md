# Minecraft-AI

The goal of this project is to apply reseach papers on a Minecraft bot to resolve different tasks.

AI realized:
- Q-Learning : the bot has to reach a blue block without falling in the lava. The bot takes as input his current position. With this, he has to choose an action and wait for the reward. The maze stay the same during all the training and test.
- Deep Q-Learning : the bot has to reach a blue block in a map which changes at each try. He should not fall in the lava. The input is now the image seen by the bot. The AI is implemented with replay memory and eligibility trace.

The next steps of this project will be to apply a World Model and A3C on the same environment to compare the results.

```bash
>> cd C:\Users\avillemin\Pictures\Malmo-0.36.0-Windows-64bit_withBoost_Python3.6\Minecraft
>> launchClient.bat
```

Environment used : https://github.com/Microsoft/malmo

# Reinforcement Learning

[Source] https://spinningup.openai.com/en/latest/index.html

1. [Reinforcement Learning Introduction](#RL)   
   a. [The Off-Policy Algorithms](#off)   
   b. [The On-Policy Algorithms](#on)   
   c. [Bellman Equations](#Bellman)     
   d. [Kinds of RL Algorithms](#kinds)   
   e. [Model-Free vs Model-Based RL](#models)   
2. [Simplest Policy Gradient](#simplest)     
3. [Reward-to-Go Policy Gradient](#togo)   
4. [Baselines in Policy Gradients](#baselines) 
5. [Other Forms of the Policy Gradient](#others)
6. [Vanilla Policy Gradient](#vanilla)
7. [Deep Q-Learning](#DQN)
8. [Asynchronous Advantage Actor-Critic](#a3c)
9. [World Models](#world)

<a name="RL"></a>

<a name="off"></a>
## The Off-Policy Algorithms
   
Algorithms like DDPG and Q-Learning are off-policy, so they are able to reuse old data very efficiently. They gain this benefit by exploiting Bellman’s equations for optimality, which a Q-function can be trained to satisfy using any environment interaction data (as long as there’s enough experience from the high-reward areas in the environment).   
<a name="on"></a>   
## The On-Policy Algorithms   
   
They don’t use old data, which makes them weaker on sample efficiency. But this is for a good reason: these algorithms directly optimize the objective you care about—policy performance—and it works out mathematically that you need on-policy data to calculate the updates. So, this family of algorithms trades off sample efficiency in favor of stability—but you can see the progression of techniques (from VPG to TRPO to PPO) working to make up the deficit on sample efficiency.   
<a name="Bellman"></a>   
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

<a name="kinds"></a>    
# Kinds of RL Algorithms

![Alt text](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)

<a name="models"></a> 
## Model-Free vs Model-Based RL

   One of the most important branching points in an RL algorithm is the question of whether the agent has access to (or learns) a model of the environment. By a model of the environment, we mean a function which predicts state transitions and rewards.   
   The main upside to having a model is that it allows the agent to plan by thinking ahead, seeing what would happen for a range of possible choices, and explicitly deciding between its options.    
   Algorithms which use a model are called model-based methods, and those that don’t are called model-free. While model-free methods forego the potential gains in sample efficiency from using a model, they tend to be easier to implement and tune. As of the time of writing this introduction (September 2018), model-free methods are more popular and have been more extensively developed and tested than model-based methods.   

What to learn:
- policies, either stochastic or deterministic,
- action-value functions (Q-functions),
- value functions,
- and/or environment models.
<a name="simplest"></a> 
# Simplest Policy Gradient   
   
We have our policy π that has a parameter θ. This π outputs a probability distribution of actions.   
   
<p align="center"><img src="https://cdn-images-1.medium.com/max/1000/0*354cfoILK19WFTWa." width="400"></p>

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

**Disadvantages**:Naturally, Policy gradients have one big disadvantage. A lot of the time, they converge on a local maximum rather than on the global optimum.   
Instead of Deep Q-Learning, which always tries to reach the maximum, policy gradients converge slower, step by step. They can take longer to train.   

Here, we consider the case of a stochastic, parameterized policy, ![](https://spinningup.openai.com/en/latest/_images/math/80088cfe6126980142c5447a9cb12f69ee7fa333.svg). We aim to maximize the expected return ![Alt Text](https://spinningup.openai.com/en/latest/_images/math/48ffbf0dd0274a46574e145ea23e4c174f6dfaa3.svg). For the purposes of this derivation, we’ll take R(tau) to give the finite-horizon undiscounted return, but the derivation for the infinite-horizon discounted return setting is almost identical.
   
There are two steps:   
- Measure the quality of a π (policy) with a policy score function J(θ)   
- Use policy gradient ascent to find the best parameter θ that improves our π.   
   
We would like to optimize the policy by gradient ascent, eg
<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/237c86938ce2e9de91040e4090f79c6a1125fc00.svg"></p> 
This is an expectation, which means that we can estimate it with a sample mean. If we collect a set of trajectories where each trajectory is obtained by letting the agent act in the environment using the policy, the policy gradient can be estimated with
<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/a8ec906d99c7cb540ef0df80d86fa1bca0f33a79.svg"></p> 

<a name="togo"></a>
# Reward-to-Go Policy Gradient   
   
In the method above, I took the sum of all rewards ever obtained. But this doesn’t make much sense.   
   
Agents should really only reinforce actions on the basis of their consequences. Rewards obtained before taking an action have no bearing on how good that action was: only rewards that come after.   
   
It turns out that this intuition shows up in the math, and we can show that the policy gradient can also be expressed by   
   
<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/9926209f608ff0134af57f8b5fa4ecf0ea515480.svg"></p>  

In this form, actions are only reinforced based on rewards obtained after they are taken. We’ll call this form the **“reward-to-go policy gradient”**, because the sum of rewards after a point in a trajectory.   
   
But how is this better? A key problem with policy gradients is how many sample trajectories are needed to get a low-variance sample estimate for them. The formula we started with included terms for reinforcing actions proportional to past rewards, all of which had zero mean, but nonzero variance: as a result, they would just add noise to sample estimates of the policy gradient. By removing them, we reduce the number of sample trajectories needed.   
   
An (optional) proof of this claim can be found here, and it ultimately depends on the EGLP (Expected Grad-Log-Prob) lemma. Suppose that P{theta} is a parameterized probability distribution over a random variable, x. Then:
<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/458b0eb0829ecd27ff745f9329fdc0fbd56295bf.svg"></p>  

<a name="baselines"></a>   
# Baselines in Policy Gradients   
   
An immediate consequence of the EGLP lemma is that for any function b which only depends on state,
<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/18738f8112fa4138e032a4be06e898543e587346.svg"></p>
   
This allows us to add or subtract any number of terms like this from our expression for the policy gradient, without changing it in expectation:
<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/a68f53bd1391212d17d426dafeb71a39c10c9189.svg"></p>
   
Any function b used in this way is called a baseline.   
   
The most common choice of baseline is the on-policy value function ![V](https://spinningup.openai.com/en/latest/_images/math/d1a28691e5690a9f6b7bce161b03bd1fc8eadbd8.svg). Recall that this is the average return an agent gets if it starts in state s_t and then acts according to policy π for the rest of its life.   
   
Empirically, the choice ![b](https://spinningup.openai.com/en/latest/_images/math/2221a29de2953ebb8930423425ed4d0feea27b25.svg) has the desirable effect of reducing variance in the sample estimate for the policy gradient. This results in faster and more stable policy learning. It is also appealing from a conceptual angle: it encodes the intuition that if an agent gets what it expected, it should “feel” neutral about it.   
   
In practice, ![](https://spinningup.openai.com/en/latest/_images/math/d1a28691e5690a9f6b7bce161b03bd1fc8eadbd8.svg) cannot be computed exactly, so it has to be approximated. This is usually done with a neural network, ![](https://spinningup.openai.com/en/latest/_images/math/a133925364fc62de281e792d4a41fbc9c360967f.svg), which is updated concurrently with the policy (so that the value network always approximates the value function of the most recent policy).
   
The simplest method for learning V_{\phi}, used in most implementations of policy optimization algorithms (including VPG, TRPO, PPO, and A2C), is to minimize a mean-squared-error objective:   

<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/e72767ea0e9d0fba2e5ab7af60aad01af4cdf0e0.svg"></p>   

where π_k is the policy at epoch k. This is done with one or more steps of gradient descent, starting from the previous value parameters ![](https://spinningup.openai.com/en/latest/_images/math/7681f537ec31c9bc1c811dd0f33f46aa9aaabebf.svg).

<a name="others"></a>
# Other Forms of the Policy Gradient

What we have seen so far is that the policy gradient has the general form
<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/1485ca5baaa09ed99fbcc54ba600e36852afd36c.svg"></p>   
   
where Phi_t could be any of the finite-horizon undiscounted return, the reward-to-go or the reward-to-go with a baseline b(s_t).   
   
All of these choices lead to the same expected value for the policy gradient, despite having different variances. It turns out that there are two more valid choices of weights \Phi_t which are important to know.     
   
**1. On-Policy Action-Value Function.** The choice ![](https://spinningup.openai.com/en/latest/_images/math/bf6f5680c7568790c744baf59bbc27831603f200.svg) is also valid. 
   
**2. The Advantage Function.** Recall that the advantage of an action describes how much better or worse it is than other actions on average (relative to the current policy). This choice, ![](https://spinningup.openai.com/en/latest/_images/math/06e42f4a5a133c3a56d70aaa098c23c3f0a37df2.svg) is also valid.   
   
The formulation of policy gradients with advantage functions is extremely common, and there are many different ways of estimating the advantage function used by different algorithms.   
<a name="vanilla"></a>
# Vanilla Policy Gradient

- VPG is an on-policy algorithm.
- VPG can be used for environments with either discrete or continuous action spaces.   
   
<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/291ee3741c3153bf4c21de62cee4823a4af59f58.svg"></p>   
     
VPG trains a stochastic policy in an on-policy way. This means that it explores by sampling actions according to the latest version of its stochastic policy. The amount of randomness in action selection depends on both initial conditions and the training procedure. Over the course of training, the policy typically becomes progressively less random, as the update rule encourages it to exploit rewards that it has already found. This may cause the policy to get trapped in local optima.   
  
<p align="center"><img src="https://spinningup.openai.com/en/latest/_images/math/47a7bd5139a29bc2d2dc85cef12bba4b07b1e831.svg"></p>

<a name="DQN"></a>   
# Deep Q-Learning
   
https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/   
   
Let’s define a function Q(s, a) such that for given state s and action a it returns an estimate of a total reward we would achieve starting at this state, taking the action and then following some policy.

<p align="center"><img src="https://s0.wp.com/latex.php?latex=Q%28s%2C+a%29+%3D+r+%2B+%5Cgamma+max_a+Q%28s%27%2C+a%29&bg=ffffff&fg=242424&s=0&zoom=2"></p>

**Experience replay**: The key idea of experience replay is that we store these transitions in our memory and during each learning step, sample a random batch and perform a gradient descend on it. Lastly, because our memory is finite, we can typically store only a limited number of samples. Because of this, after reaching the memory capacity we will simply discard the oldest sample.

**Exploration**:  A simple technique to resolve this is called ε-greedy policy. This policy behaves greedily most of the time, but chooses a random action with probability ε.   
   
**Target Network**: As a cat chasing its own tale, the network sets itself its targets and follows them. As you can imagine, this can lead to instabilities, oscillations or divergence. To overcome this problem, researches proposed to use a separate target network for setting the targets. This network is a mere copy of the previous network, but frozen in time. It provides stable Q~ values and allows the algorithm to converge to the specified target:    
![](https://s0.wp.com/latex.php?latex=Q%28s%2C+a%29+%5Cxrightarrow%7B%7D+r+%2B+%5Cgamma+max_a+%5Ctilde%7BQ%7D%28s%27%2C+a%29&bg=ffffff&fg=242424&s=0&zoom=2)      
   
After severals steps, the target network is updated, just by copying the weights from the current network. To be effective, the interval between updates has to be large enough to leave enough time for the original network to converge.   
A drawback is that it substantially slows down the learning process. Any change in the Q function is propagated only after the target network update. The intervals between updated are usually in order of thousands of steps, so this can really slow things down.

<p align="center"><img src="https://jaromiru.files.wordpress.com/2016/10/cartpole_target_vs_single_2.png?w=700&zoom=2"></p>
   
The version with target network smoothly aim for the true value whereas the simple Q-network shows some oscillations and difficulties. Although sacrificing speed of learning, this added stability allows the algorithm to learn correct behaviour in much complicated environments, such as those described in the original paper – playing Atari games receiving only visual input.

Full DQN = Q-Learning with target network and error clipping.

**Double Learning**: One problem in the DQN algorithm is that the agent tends to overestimate the Q function value, due to the max in the formula used to set targets. A solution to this problem was proposed by Hado van Hasselt (2010) and called Double Learning. In this new algorithm, two Q functions – Q_1 and Q_2 – are independently learned. One function is then used to determine the maximizing action and second to estimate its value. Either Q_1 or Q_2 is updated randomly with a formula:

<p align="center"><img src="https://s0.wp.com/latex.php?latex=Q_1%28s%2C+a%29+%5Cxrightarrow%7B%7D+r+%2B+%5Cgamma+Q_2%28s%27%2C+argmax_a+Q_1%28s%27%2C+a%29%29+&bg=ffffff&fg=242424&s=0&zoom=2"></p>
<p align="center"><img src="https://s0.wp.com/latex.php?latex=Q_2%28s%2C+a%29+%5Cxrightarrow%7B%7D+r+%2B+%5Cgamma+Q_1%28s%27%2C+argmax_a+Q_2%28s%27%2C+a%29%29+&bg=ffffff&fg=242424&s=0&zoom=2"></p>
   
It was proven that by decoupling the maximizing action from its value in this way, one can indeed eliminate the maximization bias. The Deep Reinforcement Learning with Double Q-learning paper reports that although Double DQN (DDQN) does not always improve performance, it substantially benefits the stability of learning. This improved stability directly translates to ability to learn much complicated tasks.   
When testing DDQN on 49 Atari games, it achieved about twice the average score of DQN with the same hyperparameters. With tuned hyperparameters, DDQN achieved almost four time the average score of DQN.   

**Prioritized Experience Replay**: The main idea is that we prefer transitions that does not fit well to our current estimate of the Q function, because these are the transitions that we can learn most from. This reflects a simple intuition from our real world – if we encounter a situation that really differs from our expectation, we think about it over and over and change our model until it fits.   
See: https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/   
We can define an error of a sample S = (s, a, r, s’) as a distance between the Q(s, a) and its target T(S). We will store this error in the agent’s memory along with every sample and update it with each learning step. We will then tanslate this error to a probability of being chosen for replay. Then, we create a binary tree which will be use to sample efficently our memory.  
   
Tests performed on 49 Atari games showed that PER really translates into faster learning and higher performance3. What’s more, it’s complementary to DDQN.
DQN = 100%, DQN+PER = 291%, DDQN = 343%, DDQ+PER = 451%   
An implementation of DDQN+PER for an Atari game Seaquest is available here: https://github.com/jaromiru/AI-blog/blob/master/Seaquest-DDQN-PER.py

![Eligibility](https://github.com/avillemin/Minecraft-AI/blob/master/img/eligibilityTrace.PNG)   
   
![](https://github.com/avillemin/Minecraft-AI/blob/master/img/result%20nstep%20learning.PNG)  
   
![](https://github.com/avillemin/Minecraft-AI/blob/master/img/n-step%20TD.PNG)   

n-step Sarsa can be seen as a on-policy n-step Q-learning

<a name="a3c"></a>
# Asynchronous Advantage Actor-Critic
  
https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/   
First, agent’s actions are determined by a stochastic policy π(s). Stochastic policy means that it does not output a single action, but a distribution of probabilities over actions, which sum to 1.0. We’ll also use a notation π(a|s) which means the probability of taking action a in state s. For clarity, note that there is no concept of greedy policy in this case. The policy π does not maximize any value. It is simply a function of a state s, returning probabilities for all possible actions.    
Our neural network with weights θ will now take an state s as an input and output an action probability distribution, π_θ.   
In practice, we can take an action according to this distribution or simply take the action with the highest probability, both approaches have their pros and cons.
   
But we want the policy to get better, so how do we optimize it? First, we need some metric that will tell us how good a policy is. Let’s define a function J(π) as a discounted reward that a policy π can gain, averaged over all possible starting states s_0. What we truly care about is how to improve this quantity. If we knew the gradient of this function, it would be trivial. Surprisingly, it turns out that there’s easily computable gradient of J(π) function in the following form:

<p align="center"><img src="https://s0.wp.com/latex.php?latex=%5Cnabla_%5Ctheta%5C%3BJ%28%5Cpi%29+%3D+E_%7Bs%5Csim%5Crho%5E%5Cpi%2C%5C%3Ba%5Csim%7B%5Cpi%28s%29%7D%7D%5B+A%28s%2C+a%29+%5Ccdot+%5Cnabla_%5Ctheta%5C%3Blog%5C%3B%5Cpi%28a%7Cs%29+%5D+&bg=ffffff&fg=242424&s=0&zoom=2"></p>

**Actor-Critic**: One thing that remains to be explained is how we compute the A(s, a) term.

<p align="center"><img src="https://s0.wp.com/latex.php?latex=A%28s%2C+a%29+%3D+Q%28s%2C+a%29+-+V%28s%29+%3D+r+%2B+%5Cgamma+V%28s%27%29+-+V%28s%29+&bg=ffffff&fg=242424&s=0&zoom=2"></p>

We can see that it is sufficient to know the value function V(s) to compute A(s, a). The value function can also be approximated by a neural network, just as we did with action-value function in DQN. Compared to that, it’s easier to learn, because there is only one value for each state.
  
What’s more, we can use the same neural network for estimating π(s) to estimate V(s). This has multiple benefits. Because we optimize both of these goals together, we learn much faster and effectively. 

<p align="center"><img src="https://jaromiru.files.wordpress.com/2017/02/a3c_nn_2.png?w=280&zoom=2"></p>
   
So we have two different concepts working together. The goal of the first one is to optimize the policy, so it performs better. This part is called **actor**. The second is trying to estimate the value function, to make it more precise. That is called **critic**.

**Asynchronous**: The samples we gather during a run of an agent are highly correlated. If we use them as they arrive, we quickly run into issues of online learning. In DQN, we used Experience Replay. But there’s another way to break this correlation while still using online learning. We can run several agents in parallel, each with its own copy of the environment, and use their samples as they arrive. Another benefit is that this approach needs much less memory, because we don’t need to store the samples.
   
Multiple separate environments are run in parallel, each of which contains an agent. The agents however share one neural network. Samples produced by agents are gathered in a queue, from where they are asynchronously used by a separate optimizer thread to improve the policy. 

**N-step return**: Usually we used something called 1-step return when we computed Q(s, a), V(s) or A(s, a) functions. That means that we looked only one step ahead. However, we can use more steps to give us another approximation:

<p align="center"><img src="https://s0.wp.com/latex.php?latex=V%28s_0%29+%5Cxrightarrow%7B%7D+r_0+%2B+%5Cgamma+r_1+%2B+...+%2B+%5Cgamma%5En+V%28s_n%29&bg=ffffff&fg=242424&s=0&zoom=2"></p>

The n-step return has an advantage that changes in the approximated function get propagated much more quickly. Let’s say that the agent experienced a transition with unexpected reward. In 1-step return scenario, the value function would only change slowly one step backwards with each iteration. In n-step return however, the change is propagated n steps backwards each iteration, thus much quicker.   
   
N-step return has its drawbacks. It’s higher variance because the value depends on a chain of actions which can lead into many different states. This might endanger the convergence.   
   
Full commented implementation: https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/   
https://github.com/jaromiru/AI-blog/blob/master/CartPole-A3C.py   

**Loss Function**: Now we have to define a loss function, which has three parts:

<p align="center"><img src="https://s0.wp.com/latex.php?latex=L+%3D+L_%7B%5Cpi%7D+%2B+c_v+L_v+%2B+c_%7Breg%7D+L_%7Breg%7D&bg=ffffff&fg=242424&s=0&zoom=2"></p>

L_π is the loss of the policy, L_v is the value error and L_reg is a regularization term. These parts are multiplied by constants c_v and c_reg, which determine what part we stress more.

**Policy Loss**:

<p align="center"><img src="https://s0.wp.com/latex.php?latex=L_%5Cpi+%3D+-+%5Cfrac%7B1%7D%7Bn%7D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Cunderline%7BA%28s_i%2C+a_i%29%7D+%5Ccdot+log%5C%3B%5Cpi%28a_i%7Cs_i%29+&bg=ffffff&fg=242424&s=0&zoom=2"></p>

 We need to take care to treat the advantage function as constant, by using tf.stop_gradient() operator.    
 loss_policy = - logp * tf.stop_gradient(advantage) 

**Value Loss**:

<p align="center"><img src="https://s0.wp.com/latex.php?latex=V%28s_0%29+%3D+r_0+%2B+%5Cgamma+r_1+%2B+%5Cgamma%5E2+r_2+%2B+...+%2B+%5Cgamma%5E%7Bn-1%7D+r_%7Bn-1%7D+%2B+%5Cgamma%5En+V%28s_n%29&bg=ffffff&fg=242424&s=0&zoom=2"></p>

The approximated V(s) should converge according to this formula and we can measure the error as:

<p align="center"><img src="https://s0.wp.com/latex.php?latex=e+%3D+r_0+%2B+%5Cgamma+r_1+%2B+%5Cgamma%5E2+r_2+%2B+...+%2B+%5Cgamma%5E%7Bn-1%7D+r_%7Bn-1%7D+%2B+%5Cgamma%5En+V%28s_n%29+-+V%28s_0%29&bg=ffffff&fg=242424&s=0&zoom=2"></p>

Then we can define the L_v as a mean squared error of all given samples as:

<p align="center"><img src="https://s0.wp.com/latex.php?latex=L_V+%3D+%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+e_i%5E2+&bg=ffffff&fg=242424&s=0&zoom=2"></p>

**Regularization with policy entropy**: Adding entropy to the loss function was found to improve exploration by limiting the premature convergence to suboptimal policy. Entropy for policy π(s) is defined as:

<p align="center"><img src="https://s0.wp.com/latex.php?latex=H%28%5Cpi%28s%29%29+%3D+-+%5Csum_%7Bk%3D1%7D%5E%7Bn%7D+%5Cpi%28s%29_k+%5Ccdot+log%5C%3B%5Cpi%28s%29_k&bg=ffffff&fg=242424&s=0&zoom=2"></p>

Where π(s)_k is a probability for k-th action in state s. It’s useful to know that entropy for fully deterministic policy (e.g. [1, 0, 0, 0] for four actions) is 0 and it is maximized for totally uniform policy (e.g. [0.25, 0.25, 0.25, 0.25]). Knowing this we see that by trying to maximize the entropy, we are keeping the policy away from the deterministic one. This fact stimulate exploration.   
Averaging over all samples in a batch, L_{reg} is then set to:  
<p align="center"><img src="https://s0.wp.com/latex.php?latex=L_%7Breg%7D+%3D+-+%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+H%28%5Cpi%28s_i%29%29&bg=ffffff&fg=242424&s=0&zoom=2"></p>

In the tutorial, they took c_v = 0.5 and C_reg = 0.01
<a name="world"></a>
# World Models

My dedicated repository: https://github.com/avillemin/SuperDataScience-Courses/tree/master/Hybrid%20AI

<p align="center"><img src="https://camo.githubusercontent.com/2ba6231c44dc4871d5ed65ed7ddf4b8a1e03a91b/68747470733a2f2f776f726c646d6f64656c732e6769746875622e696f2f6173736574732f776f726c645f6d6f64656c5f736368656d617469632e737667"></p>
