import os
import pickle
import random
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
    self.save_model = save_model
    self.use_pretrained_model=True
    self.model_location = 'saved_model'
    if self.use_pretrained_model:    
        self.load_model = load_model
        self.q_net = self.load_model(self,model_location = 'saved_model')
        self.target_q_net = self.load_model(self,model_location = 'saved_model')
    else:
        self.q_net = build_q_network(learning_rate=0.01)
        self.target_q_net = build_q_network(learning_rate=0.01)
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
    
    randomProb = round(max(0.9 - (state["round"])/100, 0.05), 2)
 
    if self.train == True: 
    
        if np.random.rand()<randomProb:
            return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN','WAIT', 'BOMB'], p=[.2, .2, .2, .2,.175,.025])
        
        else:
            state_input = self.reshape_game_state(state)
            action_q = self.q_net(state_input)  
            action_index = np.argmax(action_q.numpy()[0], axis=0)
            return self.actions[action_index]

    else:
        
        state_input = self.reshape_game_state(state)
        action_q = self.q_net(state_input)  
        action_index = np.argmax(action_q.numpy()[0], axis=0)
        return self.actions[action_index]
