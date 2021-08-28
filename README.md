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

## Approach


### Step 1. Data Exchange Between the Client and Game

#### ```from gym_torcs import TorcsEnv``` import gym_torcs library which is used to setup connection.
#### ```env = TorcsEnv(vision=False, throttle=True,gear_change=False)``` setup TORCS environment. 
#### ```ob = env.reset()```
#### ```s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))``` retrieves data(states) from game server.
#### ```ob, r_t, done, info = env.step(action)``` sends command(actions to be taken) to the game server, where r_t is the reward for taking that action.

### Step 2. Actor-Critic Models

#### Actor Model
##### ```def create_actor_model(state_size) :```
##### ```  S = Input(shape=[state_size])```   
##### ```  h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)```
##### ```  h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)```
##### ```  Steering = Dense(1,activation='tanh')(h1)```  
##### ```  Acceleration = Dense(1,activation='sigmoid')(h1)```   
##### ```  Brake = Dense(1,activation='sigmoid')(h1)```
##### ```  V = concatenate([Steering,Acceleration,Brake])```        
##### ```  model = Model(inputs=S,outputs=V)```
##### ```  return model```

### Step 3. WebApp Framework


![repo7](https://user-images.githubusercontent.com/64823050/129591794-b4fe2d45-27bf-4167-9be8-147a05c29cf7.jpg)



## Web Application
 
 
![repo15](https://user-images.githubusercontent.com/64823050/130605735-ca553035-4ff5-4450-9f69-431d8c5e3597.jpg)



## Result


![repo16](https://user-images.githubusercontent.com/64823050/130605750-10311cbf-d5df-4b1d-80fa-916bea1a8683.jpg)



## References


[HOG Tranformation](https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/)

