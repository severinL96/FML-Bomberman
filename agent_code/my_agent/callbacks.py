import os
import pickle
import random
import tensorflow as tf
from tensorflow.keras.layers import (Input,Dense,Lambda)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
print(os.getcwd())
from .callbacks_helper import *


actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    '''
    sets up inital q_net and target_q_net
    store possible actions
    '''
    self.reshape_game_state = reshape_game_state
    self.q_net = build_q_network(learning_rate=0.001)
    self.target_q_net = build_q_network(learning_rate=0.001)
    self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def act(self,state):
    '''
    Query the Q-Net for an action given a state (can be random action)
    Arguments: 
        state: state tp give action for
        ACTIONS: possible actions
        epsilon: The probability of doing a random move
    Returns:
        a string returning the predicted move
    '''
    if np.random.rand()<0.1:
        return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN','WAIT', 'BOMB'], p=[.2, .2, .2, .2,.1,.1])
    else:
        state_input = self.reshape_game_state(state)
        action_q = self.q_net(state_input)  
        action_index = np.argmax(action_q.numpy()[0], axis=0)
        return self.actions[action_index]



