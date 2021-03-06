import os
import random
import numpy as np
from tensorflow.keras import models
from .callbacks_helper import *

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.random_prob = True
    self.load_model = None #'random_train'

    if self.load_model is not None:
        pass
        self.logger.info("loading model "+self.load_model)
        self.q_net = models.load_model(self.load_model)
        self.target_q_net = models.load_model(self.load_model)
    else:
        self.logger.info("creating new model.")
        self.q_net = build_q_network(learning_rate=10-5)
        self.target_q_net = self.q_net
        
def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    if self.random_prob:
        random_prob = 0.3

    if self.train and random.random() <= random_prob:
        #self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.22, .22, .22, .22, .06, .06])


    state_vector = state_to_vector(game_state)
    action_q = self.q_net(state_vector).numpy()[0]    
    #self.(ACTIONS[np.argmax(action_q)],np.argmax(action_q))
    return ACTIONS[np.argmax(action_q)]



