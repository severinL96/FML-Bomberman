import pickle
import random
from typing import List
import os
import events as e
from .callbacks import state_to_map
from .train_helper import do_training, reward_from_events, do_training_with_PER_2

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.transitions = []
    self.save_location = './saved_models/CNN_first_try'
    
    if not os.path.isdir(self.save_location):
        os.mkdir(self.save_location)
    with open(self.save_location+"/loss.txt", 'w') as file: 
        file.truncate(0)
        file.close()
        


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events):
    """
    appends old_game_state,action (as an index!),reward,new_game_state to self.transitions
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    #store a static state as first move
    if new_game_state['step']<=1:
        old_game_state = new_game_state
        reward = 0
        action = None
    else:
        stateCopy= old_game_state.copy()
        old_game_state = state_to_map(old_game_state)
        reward = reward_from_events(self,events)
        action = ACTIONS.index(self_action)
        new_game_state = state_to_map(new_game_state)
        self.transitions.append([old_game_state,action,reward,new_game_state])

    # train the model and update the target q net

    
    try:
    
        if self.old_game_state["step"]%50==0:
            do_training_with_PER_2(self)
    except:
        pass
        
    
    
    
    #do_training(self)
    #self.target_q_net.set_weights(self.q_net.get_weights())

        
        
def end_of_round(self, last_game_state: dict, last_action: str, events):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    #store the last state
    last_state_vector = state_to_map(last_game_state)
    reward = reward_from_events(self,events)
    action = 0
    try: 
        action = ACTIONS.index(last_action)
    except: 
        pass
    self.transitions.append([last_state_vector,action,reward,last_state_vector])

    # train the model and update the target q net
  
    #do_training(self)
    do_training_with_PER_2(self)
    self.target_q_net.set_weights(self.q_net.get_weights())
    self.q_net.save(self.save_location)


