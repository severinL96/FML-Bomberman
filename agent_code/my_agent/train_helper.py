import numpy as np
import tensorflow as tf
import events as e
############################################################
# model related stuff
############################################################

def update_target_q_net(self):
    self.target_q_net.set_weights(self.q_net.get_weights())

############################################################
# train the model
############################################################

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
        e.INVALID_ACTION: -5, # encourages to not be stupid
        e.GOT_KILLED: -10,
        e.KILLED_SELF: -10
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
   
    return reward_sum
