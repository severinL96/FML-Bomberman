import tensorflow as tf
from tensorflow.keras.layers import (Input,Dense,Lambda)
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
    model_input = Input(shape=(1445,),name='input')

    x = Dense(128, activation='relu',name='layer0')(model_input)
    x = Dense(64, activation='relu',name='layer1')(x)
    x = Dense(32, activation='relu',name='layer2')(x)
    x = Dense(16, activation='relu',name='layer3')(x)
    
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
    q_net.add(Dense(64, input_dim=1445, activation='relu', kernel_initializer='he_uniform'))
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
    OUTPUT: all relevant game info, in the shape of [field_size,field_size] or as stacked vector of length 5*field_size**2
    
    Function reshapes all information in game state dictionary and 
    returns info in the form of maps or as unravelled stacked vector
    '''
    #extract all data from coin, bomb and players to maps
    coin_player_map = np.zeros((17,17))
    for coin_coord in game_state['coins']:
        coin_map[coin_coord]=1

    player_map = np.zeros((17,17))
    for name,score,bomb,coord in game_state['others']:
        if bomb:
            coin_player_map[coord]= -1
        else:
            coin_player_map[coord]= -0.5

    name,score,bomb,coord = game_state['self']
    if bomb:
        coin_player_map[coord]= 1
    else:
        coin_player_map[coord]= 0.5

    field_bomb_map = game_state['field']
    for bomb_coord,bomb_time in game_state['bombs']:
        field_bomb_map[bomb_coord]=bomb_time
        

    #join all maps
    state = np.stack([field_bomb_map,game_state['explosion_map'],coin_player_map])
    if vector:
        state = state.reshape(-1)
    
    state = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
    return state