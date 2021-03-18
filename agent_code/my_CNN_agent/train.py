import numpy as np
import random
import os
import events as e
from .callbacks import state_to_map
from .train_helper import do_training, reward_from_events#, do_training_with_PER_2

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
    self.rewards_in_game = []
    
    if not os.path.isdir(self.save_location):
        os.mkdir(self.save_location)
    with open(self.save_location+"/loss.txt", 'w') as file: 
        file.truncate(0)
        file.close()
    with open(self.save_location+"/average_reward.txt", 'w') as file: 
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

        
        old_game_state = state_to_map(old_game_state)
        reward = reward_from_events(self,events)
        action = ACTIONS.index(self_action)
        new_game_state = state_to_map(new_game_state)
        self.transitions.append([old_game_state,action,reward,new_game_state])
    
    self.rewards_in_game.append(reward)
        

 
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
    action = ACTIONS.index(last_action)
    #self.transitions.append([last_state_vector,action,reward,None])
        
    #do_training(self)
    
    if last_game_state["round"] % self.train_after_episodes == 0:
        # train the model and update the target q net
        do_training(self)
        
    if last_game_state["round"] % self.train_after_episodes == 0:
        # train the model and update the target q net
        self.target_q_net.set_weights(self.q_net.get_weights())
        self.q_net.save(self.save_location)

    self.rewards_in_game.append(reward)

    with open(self.save_location + "/average_reward.txt", 'a') as file: 
        try:
            average_reward = np.sum(self.rewards_in_game)/len(self.rewards_in_game)
            file.write(str(average_reward)+"\n")
        except:
            file.write(str(np.nan)+"\n")
            self.rewards_in_game = []


