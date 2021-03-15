import os
import random
import numpy as np
from tensorflow.keras import models
from .callbacks_helper import *

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np

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
    #self.load_model = './saved_models/CNN_first_try'
    #self.save_location = './saved_models/CNN_first_try'
    self.load_model = './saved_models/CNN_first_try'
    self.learning_rate = 10e-4
    if self.load_model is not None:
        pass
        self.logger.info("loading model "+self.load_model)
        self.q_net = models.load_model(self.load_model)
        self.target_q_net = models.load_model(self.load_model)
    else:
        self.logger.info("creating new model.")
        self.q_net = build_q_network(learning_rate=self.learning_rate)
        self.target_q_net = build_q_network(self.learning_rate)

def build_q_network(learning_rate): 
    """
    Builds a deep neural net which predicts the Q values for all possible
    actions given a state. The input should have the shape of the state
    (which is 4 in CartPole), and the output should have the same shape as
    the action space (which is 2 in CartPole) since we want 1 Q value per
    possible action.
    
    :return: the Q network
    """
    q_net = models.Sequential()

    q_net.add(Conv2D(16, (5, 5), activation='sigmoid', input_shape=(17,17,1)))
    q_net.add(MaxPooling2D((2, 2)))
    q_net.add(Conv2D(32, (3, 3), activation='sigmoid'))
    q_net.add(Flatten())
    q_net.add(Dense(128, activation='sigmoid', kernel_initializer='he_uniform'))
    q_net.add(Dense(64, activation='sigmoid', kernel_initializer='he_uniform'))
    q_net.add(Dense(32, activation='sigmoid', kernel_initializer='he_uniform'))
    q_net.add(Dense(6, activation='linear', kernel_initializer='he_uniform'))
    
    
    q_net.compile(optimizer=Adam(learning_rate=learning_rate), loss=tf.keras.losses.Huber())
    return q_net

def do_test(self):

    model = build_q_network(0.0001)
    x1=np.array([[0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]
    ,[0.4,1.,0.,0.,0.,0.,0.,0.2,0.,0.,0.,0.,0.,0.2,0.,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.2,0.4,0.,0.4,0.2,0.4,0.2,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.,0.2,0.,0.,0.,0.,0.2,0.2,0.,0.,0.,0.2,0.2,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.,0.,0.,0.,0.,0.,0.2,0.,0.,0.,0.,0.,0.,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.2,0.4,0.2,0.4]
    ,[0.4,0.,0.,0.,0.,0.2,0.,0.,0.,0.,0.,0.,0.,0.2,0.,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.2,0.,0.,0.,0.6,0.,0.,0.,0.,0.,0.,0.,0.2,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.2,0.,0.,0.,0.,0.2,0.,0.,0.,0.,0.,0.,0.,0.2,0.,0.4]
    ,[0.4,0.,0.4,0.2,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.,0.2,0.,0.,0.,0.,0.,0.2,0.,0.,0.2,0.2,0.,0.2,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.2,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.4]
    ,[0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]])

    x2=np.array([[0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]
    ,[0.4,0,0.,0.,0.,0.,0.,0.2,0.,0.,0.,0.,0.,0.2,0.,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.2,0.4,0.,0.4,0.2,0.4,0.2,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.,0.2,0.,0.,0.,0.,0.2,0.2,0.,0.,0.,0.2,0.2,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.,0.,0.,0.,0.,0.,0.2,0.,0.,0.,0.,0.,0.,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.2,0.4,0.2,0.4]
    ,[0.4,0.,0.,0.,0.,0.2,0.,0.,0.,0.,0.,0.,0.,0.2,0.,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.2,0.,0.,0.,0.6,0.,0.,0.,0.,0.,0.,0.,0.2,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.2,0.,0.,0.,0.,0.2,0.,0.,0.,0.,0.,0.,0.,0.2,0.,0.4]
    ,[0.4,0.,0.4,0.2,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.,0.2,0.,0.,0.,0.,0.,0.2,0.,0.,0.2,0.2,0.,0.2,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.2,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1,0.4]
    ,[0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]])

    x3=np.array([[0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]
    ,[0.4,0,0.,0.,0.,0.,0.,0.2,0.,0.,0.,0.,0.,0.2,0.,1,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.2,0.4,0.,0.4,0.2,0.4,0.2,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.,0.2,0.,0.,0.,0.,0.2,0.2,0.,0.,0.,0.2,0.2,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.,0.,0.,0.,0.,0.,0.2,0.,0.,0.,0.,0.,0.,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.2,0.4,0.2,0.4]
    ,[0.4,0.,0.,0.,0.,0.2,0.,0.,0.,0.,0.,0.,0.,0.2,0.,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.2,0.,0.,0.,0.6,0.,0.,0.,0.,0.,0.,0.,0.2,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.2,0.,0.,0.,0.,0.2,0.,0.,0.,0.,0.,0.,0.,0.2,0.,0.4]
    ,[0.4,0.,0.4,0.2,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.,0.2,0.,0.,0.,0.,0.,0.2,0.,0.,0.2,0.2,0.,0.2,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.2,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0,0.4]
    ,[0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]])

    x4=np.array([[0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]
    ,[0.4,0,0.,0.,0.,0.,0.,0.2,0.,0.,0.,0.,0.,0.2,0.,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.2,0.4,0.,0.4,0.2,0.4,0.2,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.,0.2,0.,0.,0.,0.,0.2,0.2,0.,0.,0.,0.2,0.2,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.,0.,0.,0.,0.,0.,0.2,0.,0.,0.,0.,0.,0.,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.2,0.4,0.2,0.4]
    ,[0.4,0.,0.,0.,0.,0.2,0.,0.,0.,0.,0.,0.,0.,0.2,0.,0.,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.2,0.,0.,0.,0.6,0.,0.,0.,0.,0.,0.,0.,0.2,0,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.2,0.,0.,0.,0.,0.2,0.,0.,0.,0.,0.,0.,0.,0.2,0.,0.4]
    ,[0.4,0.,0.4,0.2,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,0.,0.,0.2,0.,0.,0.,0.,0.,0.2,0.,0.,0.2,0.2,0.,0.2,0.4]
    ,[0.4,0.,0.4,0.,0.4,0.,0.4,0.,0.4,0.2,0.4,0.,0.4,0.,0.4,0.,0.4]
    ,[0.4,1,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.4]
    ,[0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]])


    y1 = np.array([0,1,1,0,0,0])
    y1 = np.array([0,1,1,0,0,0])
    y1 = np.array([0,1,1,0,0,0])
    y2 = np.array([1,0,0,1,0,0])

    X = []
    Y= []
    for i in range(3):
        X.append(x1)
        X.append(x2)
        Y.append(y1)
        Y.append(y2)

    X = np.expand_dims(X,axis=-1)
    X = np.array(X)
    Y = np.array(Y)


    for i in range(1000):
        if i %100 == 0:
            print(i)
        history = model.fit(X,Y,verbose = 0,epochs = 10)

    for x in [x1,x2,x3,x4]:
        pred = model(np.expand_dims(np.expand_dims(x,axis=0),axis=-1)).numpy()
        print(np.round(pred,2))

def act(self, game_state: dict) -> str:
    do_test(self)



