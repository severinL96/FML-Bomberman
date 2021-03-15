import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np




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



def state_to_map(game_state):
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

    state_map = np.expand_dims(state_map,axis=-1)
    return state_map
