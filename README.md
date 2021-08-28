# Self Driving Car in TORCS Simulator

This project was aimed to develop a self driving car agent in TORCS simulator using Deep Reinforcement Learning Actor-Critic Algorithm. 


## Dependencies

You can install Python dependencies using ``` pip install -r requirements.txt ``` , and it should just work. if you want to install package manually, here's a list:

 - Python==3.7
 - Tensorflow-gpu==2.3.0
 - Keras=2.6.0
 - Numpy=1.18.5
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

This Algorithm was implemented using tensorflow as follows :

![repo20](https://user-images.githubusercontent.com/64823050/131214556-72bb1530-9921-43cd-98e7-952f4289dff0.png)

#### ```loss = tf.convert_to_tensor(critic.train_on_batch([states,actions], y_t))``` Trained critic network on states and actions obtained from actor network, the true vaules y_t are obtained from target network.
#### ```a_for_grad = actor(states)``` Obtained actions using actor_networks by passing states in it.
#### ```qsa = critic([states,a_for_grad])``` qsa is the outpt of critic network for states, a_for_grad, which will be used for updating actor policy.
#### ```grads = tape.gradient(qsa,actor.trainable_weights)``` Calculated gradients of actor policy with respect to the output of critic network. 
#### ```opt.apply_gradients(zip(grads, actor.trainable_weights))``` Updated actor weights using gradients obtained above.
#### ```critic_target.trainable_weights[i] = 0.001*critic.trainable_weights[i] + (1-0.001)*critic.trainable_weights[i]``` Soft update of parameters of critic target network. Similarly actor target network's parameters were updated.
       


## Result

## References
