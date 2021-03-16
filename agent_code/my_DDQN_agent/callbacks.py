import os
import random
import numpy as np
from tensorflow.keras import models
from .callbacks_helper import *
from tensorflow.keras import optimizers

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']



class DDDQN(tf.keras.Model):
    def __init__(self):
      super(DDDQN, self).__init__()
      self.d1 = tf.keras.layers.Dense(128, activation='relu')
      self.d2 = tf.keras.layers.Dense(128, activation='relu')
      self.v = tf.keras.layers.Dense(1, activation=None)
      self.a = tf.keras.layers.Dense(6, activation=None)

    def call(self, input_data):
      x = self.d1(input_data)
      x = self.d2(x)
      v = self.v(x)
      a = self.a(x)
      Q = v +(a -tf.math.reduce_mean(a, axis=1, keepdims=True))
      return Q

    def advantage(self, state):
      x = self.d1(state)
      x = self.d2(x)
      a = self.a(x)
      return a


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
    self.load_model = None #'./saved_models/CNN_first_try'
    self.gamma = 0.9
    if self.load_model is not None:
        pass
        self.logger.info("loading model "+self.load_model)
        self.q_net = models.load_model(self.load_model)
        self.target_q_net = models.load_model(self.load_model)
    else:
        self.logger.info("creating new model.")
        self.q_net = DDDQN()
        self.target_q_net = DDDQN()
        self.q_net.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=10e-6))
        self.target_q_net.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=10e-6))


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = max(0.1 , 1- game_state['round']/3000)
    if self.train and random.random() <= random_prob:
        #self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.22, .22, .22, .22, .06, .06])

    state_vector = state_to_vector(game_state)
    state_vector = np.expand_dims(state_vector,axis=0)
    action_q = self.q_net.advantage(state_vector).numpy()[0]    

    return ACTIONS[np.argmax(action_q)]



