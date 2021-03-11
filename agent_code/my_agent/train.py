import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np
import tensorflow as tf
import events as e
from .callbacks import reshape_game_state
from .train_helper import *


def do_training(self):
    '''
    take a batch of experiences and update the model
    '''
  
    batch = np.array(self.transitions, dtype="object")
    oldStateBATCH = batch[:,0]
    actionBATCH = batch[:,1]
    rewardBATCH = batch[:,2]
    newStateBATCH= batch[:,3]   
      
    # Collect all oldState-expectedReward pairs     
    X=[]
    Y=[]
    for k in range(0, len(oldStateBATCH)):
        if oldStateBATCH[k] != None and newStateBATCH[k] != None:
            old_game_state_batch = oldStateBATCH[k]
            action_batch = [actionBATCH[k]]
            reward_batch = [rewardBATCH[k]]
            new_game_state_batch = newStateBATCH[k]             

            current_q = self.q_net(old_game_state_batch)
            target_q = np.copy(current_q)
            next_q = self.target_q_net(new_game_state_batch)
            max_next_q = np.amax(next_q, axis=1)

            #print("Observed Reward")
            #print(reward_batch)

            #print("ACTION")
            #print(action_batch)            
            #print("Rewards vorher")
            #print(target_q)
            
            
            
            #Calulate the expected reward
            target_q[0][action_batch[0]] = reward_batch[0] + 0.95 * max_next_q[0]
      


            #print("Rewards nachher")
            #print(target_q)
            #print("")



            X.append(old_game_state_batch)
            Y.append(target_q)   

    #print(actionBATCH)
        
    #Fit the model for all collected pairs
    if self.print_loss:
        result = self.q_net.fit(x=np.array(X)[0], y=np.array(Y)[0], verbose=0)
        loss = result.history['loss'][0]
        print('loss:' +str(loss))


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    appends old_game_state,action,reward,new_game_state to self.transitions
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    # state_to_features is defined in callbacks.py
    if new_game_state['step']<=1:
        
        old_game_state = new_game_state
        reward = 0
        action = None
    else:
        old_game_state = self.reshape_game_state(old_game_state)
        reward = self.reward_from_events(events)
        action = [i for i in range(0, len(self.actions)) if self.actions[i] == self_action][0]
        new_game_state = self.reshape_game_state(new_game_state)
        self.transitions.append([old_game_state,action,reward,new_game_state])
    

    
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    last_game_state = self.reshape_game_state(last_game_state)
    reward = self.reward_from_events(events)
    action = [i for i in range(0, len(self.actions)) if self.actions[i] == last_action][0]
    self.transitions.append([last_game_state,action,reward,None])

    self.do_training(self)
    self.update_target_q_net(self)
    self.save_model(self)

def reward_from_events(events):
    """
    Arguments:
        a list of event strings 
    Returns:
        a reward corresponding to the events' value 
    """
    game_rewards = {
        #Positive rewards
        #e.COIN_FOUND: 0.1, # encourages exploration
        e.COIN_COLLECTED: 20, 
        #e.KILLED_OPPONENT: 10,
        #e.SURVIVED_ROUND: 1, # encourages survival

        #Move penaltys
        #e.MOVED_DOWN: -0.1, # encourages efficent movement
        #e.MOVED_LEFT: -0.1,
        #e.MOVED_RIGHT: -0.1,
        #e.MOVED_UP: -0.1,
        #e.WAITED: -0.1,
        #e.BOMB_DROPPED: -0.1,

        #Stupid penaltys
        e.INVALID_ACTION: -5, # encourages to not be stupid
        #e.GOT_KILLED: -10,
        e.KILLED_SELF: -20
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
   
    return reward_sum


#Moritz: Initialize the NN here?
def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.do_training = do_training
    self.print_loss = False
    self.reward_from_events = reward_from_events
    self.transitions = []
    self.update_target_q_net = update_target_q_net
    
    print("TRAINING")
    

