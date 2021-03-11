import tensorflow as tf
from tensorflow.keras.layers import (Input,Dense)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
import numpy as np

############################################################
# init different models
############################################################
def build_difficult_q_network(learning_rate=0.00001):
    """Builds a DQN as a Keras model
    
    Arguments:
        n_actions: Number of possible action the agent can take
        learning_rate: Learning rate
        input_shape: Shape of the preprocessed frame the model sees
        history_length: Number of historical frames the agent can see
        
    Returns:
        A compiled Keras model
    """
    model_input = Input(shape=(289,),name='input')

    x = Dense(128, activation='sigmoid',name='layer0')(model_input)
    x = Dense(64, activation='sigmoid',name='layer1')(x)
    x = Dense(32, activation='sigmoid',name='layer2')(x)
    x = Dense(16, activation='sigmoid',name='layer3')(x)
    
    q_vals = Dense(6,activation='linear')(x)  #activation ='softmax')(x)

    # Build model
    model = Model(model_input, q_vals)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model


def build_q_network(learning_rate=0.001):
    """
    Builds a deep neural net which predicts the Q values for all possible
    actions given a state. The input should have the shape of the state
    (which is 4 in CartPole), and the output should have the same shape as
    the action space (which is 2 in CartPole) since we want 1 Q value per
    possible action.
    
    :return: the Q network
    """
    q_net = Sequential()
    q_net.add(Dense(64, input_dim=289, activation='relu', kernel_initializer='he_uniform'))
    q_net.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
    q_net.add(Dense(6, activation='linear', kernel_initializer='he_uniform'))
    q_net.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='mse')
    return q_net

############################################################
# load and save models
############################################################

def save_model(self):
    '''saves model at location specified in self.model_location'''
    tf.saved_model.save(self.q_net,self.model_location)

def load_model(self,model_location):
    '''loads model from model location'''
    return models.load_model(self.model_location)
############################################################
# reshape game state
############################################################


def reshape_game_state(game_state, vector=True):
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

    info_map = np.zeros((17,17))

    name,score,bomb,coord = game_state['self']
    info_map[coord]= 1
    for name,score,bomb,coord in game_state['others']:
        info_map[coord]= .8

    for coin in game_state['coins']:
        info_map[coin] = 0.6

    field = game_state['field']
    field = np.where(field == -1, .4, field) #revalue wall
    field = np.where(field == 1, .2, field) #revalue crate
    info_map = np.where(field!=0,field,info_map) #add field data to info

    explosion = - game_state['explosion_map']/4
    info_map = np.where(explosion!=0,explosion,info_map)

    print(info_map)
    for coord, timer in game_state['bombs']:
        info_map[coord] = -(timer/10 + .6)

    #join all maps
    if vector:
        info_map = info_map.reshape(-1)
    info_map = tf.convert_to_tensor(info_map[None, :], dtype=tf.float32)
    return info_map