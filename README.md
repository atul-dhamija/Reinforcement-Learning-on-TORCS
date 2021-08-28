# Self Driving Car in TORCS Simulator

This project was aimed to develop a self driving car agent in TORCS simulator using Deep Reinforcement Learning Actor-Critic Algorithm. 


## Dependencies

You can install Python dependencies using ``` pip install -r requirements.txt ``` , and it should just work. if you want to install package manually, here's a list:

 - Python==3.7
 - Tensorflow-gpu==2.3
 - Keras
 - Numpy
 - gym_torcs


## Background

TORCS simulator is an open source car simulator which is extensively used in AI research. The reason for selecting TORCS for this project is that it is easy to get states from the game using gym_torcs library, which uses SCR plugin to setup connection with the game and thus making it easy to send commands into the game and also retrieving current states. In reinforcement learning we need to get states data and send action values continuously, so this simulator suited best for our project. 
Self driving car is an area of wide research and it encompasses many fields, implementation of this project was a good method for practically applying various concepts of reinforcement learning.

## Approach

### Actor-Critic Background

### Data Exchange Between the Client and Game

#### ```from gym_torcs import TorcsEnv``` import gym_torcs library which is used to setup connection.
#### ```env = TorcsEnv(vision=False, throttle=True,gear_change=False)``` setup TORCS environment. 
#### ```ob = env.reset()```
#### ```s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))``` retrieves data(states) from game server.
#### ```ob, r_t, done, info = env.step(action)``` sends command(actions to be taken) to the game server, where r_t is the reward for taking that action.

### Actor-Critic Models

#### Actor Model

![repo17](https://user-images.githubusercontent.com/64823050/131214303-8dbdedb2-e890-4c14-8d11-9125f9d82808.png)

#### Critic Model

![repo18](https://user-images.githubusercontent.com/64823050/131214316-d3326d2f-d198-40a7-8b3e-fb05885bc183.png)

### Model Working

![repo19](https://user-images.githubusercontent.com/64823050/131214519-4b4bb198-1e77-4fd8-91c2-0a58fe5f5393.png)

This Algorithm was implemente using tensorflow as follows :

![repo20](https://user-images.githubusercontent.com/64823050/131214556-72bb1530-9921-43cd-98e7-952f4289dff0.png)


## Result


![repo16](https://user-images.githubusercontent.com/64823050/130605750-10311cbf-d5df-4b1d-80fa-916bea1a8683.jpg)



## References


[HOG Tranformation](https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/)

