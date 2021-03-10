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
    batch = np.array(self.transitions)

    oldStateBATCH = batch[:,0]
    #old_game_state_batch = tf.convert_to_tensor(old_game_state_batch[None, :], dtype=tf.float32)
    actionBATCH = batch[:,1]
    #action_batch = tf.convert_to_tensor(action_batch[None, :], dtype=tf.float32)
    rewardBATCH = batch[:,2]
    #reward_batch = tf.convert_to_tensor(reward_batch[None, :], dtype=tf.float32)
    newStateBATCH= batch[:,3]
    #new_game_state_batch = tf.convert_to_tensor(new_game_state_batch[None, :], dtype=tf.float32)    
    
    
    for k in range(0, len(oldStateBATCH)):
    
        try:

            old_game_state_batch = oldStateBATCH[k]
            action_batch = [actionBATCH[k]]
            reward_batch = [rewardBATCH[k]]
            new_game_state_batch = newStateBATCH[k]             



            print("old state")
            print(old_game_state_batch)
            print("action")
            print(action_batch)
            print("reward")
            print(reward_batch)
            print("new state")
            print(new_game_state_batch)


            #print("HERE IS THE OLD STATE")
            #print(old_game_state_batch)


            current_q = self.q_net(old_game_state_batch)
            target_q = np.copy(current_q)
            next_q = self.target_q_net(new_game_state_batch)
            max_next_q = np.amax(next_q, axis=1)


            for i in range(old_game_state_batch.shape[0]):

                target_q[i][action_batch[i]] = reward_batch[i] + 0.95 * max_next_q[i]
            result = self.q_net.fit(x=old_game_state_batch, y=target_q)


        except Exception as e:
            print(e)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    appends old_game_state,action,reward,new_game_state to self.transitions
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # state_to_features is defined in callbacks.py
    if new_game_state['step']<=1:
        old_game_state = None
        reward = 0
        action = None
        new_game_state = self.reshape_game_state(new_game_state)
    else:
        old_game_state = self.reshape_game_state(old_game_state)
        reward = self.reward_from_events(events)
        action = [i for i in range(0, len(self.actions)) if self.actions[i] == self_action][0] #returns index of action
        new_game_state = self.reshape_game_state(new_game_state)

    self.transitions.append([old_game_state,action,reward,new_game_state])



def end_of_round(self, last_game_state, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    last_game_state = self.reshape_game_state(last_game_state)
    reward = self.reward_from_events(events)

    self.transitions.append([last_game_state,last_action,reward,None])

    '''# Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
    '''
    
    self.do_training(self)


def reward_from_events(events):
    """
    Arguments:
        a list of event strings 
    Returns:
        a reward corresponding to the events' value 
    """
    game_rewards = {
        #Positive rewards
        e.COIN_FOUND: 0.1, # encourages exploration
        e.COIN_COLLECTED: 1, 
        e.KILLED_OPPONENT: 5,
        e.SURVIVED_ROUND: 5, # encourages survival

        #Move penaltys
        e.MOVED_DOWN: -0.1, # encourages efficent movement
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.MOVED_UP: -0.1,
        e.WAITED: -0.1,

        #Stupid penaltys
        e.INVALID_ACTION: -3, # encourages to not be stupid
        e.GOT_KILLED: -10,
        e.KILLED_SELF: -10
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
    self.reward_from_events = reward_from_events
    self.transitions = []
    self.update_target_q_net = update_target_q_net

