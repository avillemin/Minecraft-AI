# Minecraft-AI

The goal of this project is to apply reseach papers on a Minecraft bot to resolve different tasks.

1. [Q-Learning](#Ql)   
2. [Deep Q-Learning](#DQl)  
3. [Double Deep Q-Learning with Prioritized Experience Replay](#DDQl)   
4. [World Model](#wm) 

```bash
>> cd C:\Users\avillemin\Pictures\Malmo-0.36.0-Windows-64bit_withBoost_Python3.6\Minecraft
>> launchClient.bat
```

Environment used : https://github.com/Microsoft/malmo

<a name="Ql"></a>
# Q-Learning

<p align="center"><img src="https://github.com/avillemin/Minecraft-AI/blob/master/DQN/Qvalues.png" height="350px"></p>

<a name="DQl"></a>
# Deep Q-Learning

As expected, the result is not very good. Why? Because there is only a positive reward when the Agent reaches the final blue block. The problem is that it happens very few because the agent fell into the lava before. So the sample when the agent reaches the positive reward represents less than 1% of the samples with a random policy. As I'm using samples randomly selected from the memory to train my model, the agent doesn't learn correctly the case when the positive reward occurs. All the Q-values predicted are negative. To deal with this issue, we need to use a memory with Prioritized Experience Replay.

<a name="DDQl"></a>
# Double Deep Q-Learning with Prioritized Experience Replay

<p align="center"><img src="https://github.com/avillemin/Minecraft-AI/blob/master/DDQNPER/victory.gif" height="256px"></p>

<a name="wm"></a>
# World Model

The Minecraft environment is very heavy and the game easily runs out of memory with a long training. My approach to deal with this issue is to create a World Model. By creating a neural network able to dream and play Minecraft without the environment, we can easily improve the learning and parallelize the process. First, let's create a variational autoencoder able to encode our input images into a smaller vector:

<p align="center"><img src="https://worldmodels.github.io/assets/conv_vae_label.svg" width="350" height="600"></p>

Here is the result of the VAE with the original image and the reconstructed image:

<p align="center"><img src="https://github.com/avillemin/Minecraft-AI/blob/master/World-Model/figures/VAE.png"></p>
