# Self Driving Car in TORCS Simulator

This project was aimed to develop a self driving car agent in TORCS simulator using Deep Reinforcement Learning Actor-Critic Algorithm. 


## Dependencies

You can install Python dependencies using ``` pip install -r Requirements.txt ``` , and it should just work. if you want to install package manually, here's a list:

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

![repo22](https://user-images.githubusercontent.com/64823050/131215722-107c3db0-9d05-4c19-88dc-cb28de4e5ee1.jpg)

Imagine you play a video game with a friend that provides you some feedback. You’re the Actor and your friend is the Critic.
At the beginning, you don’t know how to play, so you try some action randomly. The Critic observes your action and provides feedback.
Learning from this feedback, you’ll update your policy and be better at playing that game.
On the other hand, your friend (Critic) will also update their own way to provide feedback so it can be better next time.
As we can see, the idea of Actor Critic is to have two neural networks. We estimate both, both run in parallel.
Because we have two models (Actor and Critic) that must be trained, it means that we have two set of weights, the weights of actor network are updated with resect toh the output of critic network. Update of target networks is done by soft update.

### Why Actor-Critic ?

![repo23](https://user-images.githubusercontent.com/64823050/131215846-7e6ed02c-b227-4990-9a3d-df0a1537a447.jpg)

The Actor Critic model is a better score function. Instead of waiting until the end of the episode as we do in Monte Carlo REINFORCE, we make an update at each step (TD Learning).
Because we do an update at each time step, we can’t use the total rewards R(t). Instead, we need to train a Critic model that approximates the value function (remember that value function calculates what is the maximum expected future reward given a state and an action). This value function replaces the reward function in policy gradient that calculates the rewards only at the end of the episode.

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
#### ```critic_target.trainable_weights[i] = 0.001*critic.trainable_weights[i] + (1-0.001)*critic.trainable_weights[i]``` Soft update of parameters of critic_target network with critic network parameters. Similarly actor_target network's parameters were updated.
       


## Result

![Hnet com-image (1)](https://user-images.githubusercontent.com/64823050/131216641-b338be42-a5cb-4160-862c-312b404621b9.gif)


## References

### [Deep-RL-Course](https://simoninithomas.github.io/deep-rl-course/)
### [RL-Blog by yanpanlau](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html)
### [Optimizing hyperparameters of deep reinforcement learning for autonomous driving based on whale optimization algorithm by Nesma M. AshrafID1, Reham R. Mostafa2, Rasha H. Sakr1, M. Z. Rashad](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0252754)
