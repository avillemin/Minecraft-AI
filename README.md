# Minecraft-AI

The goal of this project is to apply reseach papers on a Minecraft bot to resolve different tasks.

AI realized:
- Q-Learning : the bot has to reach a blue block without falling in the lava. The bot takes in input his current position. With this, he has to choose an action and wait for the reward. The maze stay the same during all the training and test.
- Deep Q-Learning : the bot has to reach a blue block in a map which changes at each try. He should not fall in the lava. The input is now only the image seen by the bot. The AI is implemented with replay memory and eligibility trace.

I used pyTorch to realize the two bots.

The file DQL_bot.py create the Minecraft environment and build the map. Then, it will realize the actions chosen by the neural network created in DQL_network.py.

The next steps of this project will be to apply Augmented Random Search and A3C on the same environment to compare the results.

Environment used : https://github.com/Microsoft/malmo

N-step Q-learning : https://papoudakis.github.io/announcements/qlearning/
Deep reinforcement learning https://arxiv.org/pdf/1312.5602.pdf
Asynchronous Methods for Deep Reinforcement Learning : https://arxiv.org/pdf/1602.01783.pdf
