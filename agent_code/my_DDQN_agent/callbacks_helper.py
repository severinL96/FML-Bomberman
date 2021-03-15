import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np



class DDDQN(tf.keras.Model):
    def __init__(self):
      super(DDDQN, self).__init__()

      self.d1 = tf.keras.layers.Dense(128, activation='relu')
      self.d2 = tf.keras.layers.Dense(128, activation='relu')
      self.v = tf.keras.layers.Dense(1, activation=None)
      self.a = tf.keras.layers.Dense(env.action_space.n, activation=None)

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

def state_to_vector(game_state):
    '''
    INPUT: game_state dict, size of field, vector = True
            game_state: {field: array(w,h),
                        bombs:[(int,int),int],
                        exp.map: array(w,h),
                        coins[(x,y)]
                        self:(str,int,bool,(int,int))
                        others:(str,int,bool,(int,int))
                        }
    OUTPUT: all relevant game info, in the shape of [field_size,field_size] or as stacked vector of length 5*field_size**2

    Function reshapes all information in game state dictionary and 
    returns info in the form of maps or as unravelled stacked vector
    '''
    if game_state is None:
        return None
    state_map = np.zeros((17,17))

    name,score,bomb,coord = game_state['self']
    state_map[coord]= 1
    for name,score,bomb,coord in game_state['others']:
        state_map[coord]= .8

    for coin in game_state['coins']:
        state_map[coin] = 0.6

    field = game_state['field']
    field = np.where(field == -1, .4, field) #revalue wall
    field = np.where(field == 1, .2, field) #revalue crate
    state_map = np.where(field!=0,field,state_map) #add field data to info

    explosion = - game_state['explosion_map']/4
    state_map = np.where(explosion!=0,explosion,state_map)

    for coord, timer in game_state['bombs']:
        state_map[coord] = -(timer/10 + .6)

    #join all maps
    state_vector = state_map.reshape(-1)

    state_vector = tf.convert_to_tensor(state_vector[None, :], dtype=tf.float32)
    return state_vector