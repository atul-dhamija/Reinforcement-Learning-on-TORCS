from gym_torcs import TorcsEnv
import random
import argparse
import json
import numpy as np
import math
import os
import tensorflow.keras.initializers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, concatenate, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
HIDDEN1_UNITS = 64
HIDDEN2_UNITS = 32
def create_actor_model(state_size) :
    S = Input(shape=[state_size])   
    h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
    h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
    Steering = Dense(1,activation='tanh')(h1)  
    Acceleration = Dense(1,activation='sigmoid')(h1)   
    Brake = Dense(1,activation='sigmoid')(h1) 
    V = concatenate([Steering,Acceleration,Brake])          
    model = Model(inputs=S,outputs=V)
    return model
def create_critic_model(state_size, action_dim) :
    S = Input(shape=[state_size])  
    A = Input(shape=[action_dim],name='action2')   
    w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
    a1 = Dense(HIDDEN2_UNITS, activation='linear')(A) 
    h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
    h2 = concatenate([h1,a1])    
    h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
    V = Dense(action_dim,activation='linear')(h3)   
    model = Model(inputs=[S,A],outputs=V)
    adam = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=adam)
    return model
from ReplayBuffer import ReplayBuffer
from OU import OU
import timeit

OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(train_indicator=1):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 29  #of sensors input

    #np.random.seed(1337)

    vision = False

    EXPLORE = 100000
    episode_count = 2000
    max_steps = 100
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0
    actor = create_actor_model(state_dim)
    actor_target = create_actor_model(state_dim)
    critic = create_critic_model(state_dim, action_dim)
    critic_target = create_critic_model(state_dim, action_dim)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)

    #Now load the weight
    print("Now we load the weight")
    try:
        actor.load_weights("actormodel.h5")
        critic.load_weights("criticmodel.h5")
        actor_target.load_weights("actormodel.h5")
        critic_target.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        print(s_t)
        total_reward = 0.
        for j in range(max_steps):
            start_time = time.process_time()
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            a_t_original = actor.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = (a_t_original[0][0] + noise_t[0][0])/4
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            ob, r_t, done, info = env.step(a_t[0])
            print(a_t[0])
            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            print(len(batch))
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])
            target_q_values = critic_target.predict([new_states, actor_target.predict(new_states)])
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
                    
       
            if (train_indicator==1):
                with tf.GradientTape() as tape :
                    opt = Adam(learning_rate=0.001)
                    loss = tf.convert_to_tensor(critic.train_on_batch([states,actions], y_t))
                    a_for_grad = actor(states)
                    qsa = critic([states,a_for_grad])
                grads = tape.gradient(qsa,actor.trainable_weights)
                opt.apply_gradients(zip(grads, actor.trainable_weights))
                for i in range(len(critic.trainable_weights)) :
                    critic_target.trainable_weights[i] = 0.001*critic.trainable_weights[i] + (1-0.001)*critic.trainable_weights[i]
                for i in range(len(actor.trainable_weights)) :
                    actor_target.trainable_weights[i] = 0.001*actor.trainable_weights[i] + (1-0.001)*actor.trainable_weights[i]
                total_reward += r_t
                s_t = s_t1
                print(time.process_time() - start_time,'time')
        
            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
        
            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.to_json(), outfile)

                critic.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
