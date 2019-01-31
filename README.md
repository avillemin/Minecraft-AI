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

<p align="center"><img src="https://github.com/avillemin/Minecraft-AI/blob/master/World-Model/figures/VAE.png"></p>
