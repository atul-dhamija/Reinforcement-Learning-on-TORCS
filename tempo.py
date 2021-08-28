from gym_torcs import TorcsEnv
import random
import argparse
import json
import numpy as np
import math
import tensorflow.keras.initializers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, concatenate, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K
from ReplayBuffer import ReplayBuffer
from OU import OU
import timeit
import time
OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(train_indicator=1):   
    env = TorcsEnv(vision=False, throttle=True,gear_change=False)
    epsilon = 1
    for i in range(10000):
        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        for j in range(500000):
            start_time = time.process_time()
            a_t_original = np.zeros([1,3])
            a_t = np.zeros([1,3])
            noise_t = np.zeros([1,3])
            a_t_original[0][0] = 0
            a_t_original[0][1] = 1
            a_t_original[0][2] = 0
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)
            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            ob, r_t, done, info = env.step(a_t[0])
            print(j,time.process_time() - start_time,'time')
    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
