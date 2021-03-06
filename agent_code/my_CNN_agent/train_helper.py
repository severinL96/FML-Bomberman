from .callbacks_helper import * 
import events as e
from tensorflow.keras.callbacks import  LearningRateScheduler
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import warnings
warnings.filterwarnings("ignore")



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']



def do_training(self):
    '''
    INPUT: Transitions
    Takes the transitions and calculates q values for the respective
    states. Trains the model to correctly predict an actions q value
    '''
    X = []
    Y = []

    for transition in self.transitions[1:-1]: # ingore first move
        old_state, action, reward, new_state = transition

        current_q = self.q_net(np.expand_dims(np.expand_dims(old_state,axis=0),axis=-1))
        target_q = np.copy(current_q)[0]
        next_q = self.target_q_net(np.expand_dims(np.expand_dims(new_state,axis=0),axis=-1))
        
        # correct the prediction for highest rewards
        target_q[action] = reward + self.gamma * np.amax(next_q)
        #target_q = np.expand_dims(target_q, axis=0)

        self.logger.debug('action: '+str(action) +' ('+ ACTIONS[action]+') got reward: '+str(reward))
        self.logger.debug('current'+str(np.array(current_q)))
        self.logger.debug('target  '+ str(target_q))

        #if abs(np.linalg.norm(current_q - target_q)) >= 1:
        X.append(old_state)
        Y.append(target_q)

    X = np.expand_dims(X,axis=-1)
    X = np.array(X)
    Y = np.array(Y)
   # print("  " + str(len(Y)))

    


    # train the model on the new data and update the target q net
    history = self.q_net.fit(X,Y,epochs = self.train_epochs, verbose = self.verbose, shuffle =True) 
    with open(self.save_location + "/loss.txt", 'a') as file: 
        for i in range(len(history.history["loss"])):
            file.write(str(history.history['loss'][i])+"\n")
            
    self.transitions = []

def do_training_with_PER_2(self):
    '''
    INPUT: Transitions
    Takes the transitions and calculates q values for the respective
    states. Trains the model to correctly predict an actions q value
    '''
    X = []
    Y = []    
    size = int(round(len(self.transitions)/5, 0))
    diff = []
    

    # Get all the errors
    for transition in self.transitions: # ingore first move
        try:
            old_state, action, reward, new_state = transition
            current_q = self.q_net(np.expand_dims(np.expand_dims(old_state,axis=0),axis=-1))
            target_q = np.copy(current_q)[0]
            next_q = self.target_q_net(np.expand_dims(np.expand_dims(new_state,axis=0),axis=-1))
            # correct the prediction for highest rewards
            target_q[action] = reward + 1 * np.amax(next_q)
            #target_q = np.expand_dims(target_q, axis=0)
            diff.append(np.linalg.norm(current_q - target_q))
        except:
            pass

        

        
    #Only keep the ones with highest error
    indices = np.argsort(diff)
    indices = indices[-size:]
    
    for transition in [self.transitions[i] for i in indices]:
        old_state, action, reward, new_state = transition

        current_q = self.q_net(np.expand_dims(np.expand_dims(old_state,axis=0),axis=-1))
        target_q = np.copy(current_q)[0]
        next_q = self.target_q_net(np.expand_dims(np.expand_dims(new_state,axis=0),axis=-1))
    
        # correct the prediction for highest rewards
        target_q[action] = reward + 1 * np.amax(next_q)
        #target_q = np.expand_dims(target_q, axis=0)
       
        self.logger.debug('action: '+str(action) +' ('+ ACTIONS[action]+') got reward: '+str(reward))
        self.logger.debug('current'+str(np.array(current_q)))
        self.logger.debug('target  '+ str(target_q))

        X.append(old_state)
        Y.append(target_q)

    X = np.expand_dims(X,axis=-1)
    X = np.array(X)
    Y = np.array(Y)
    for k in range(20):

        # train the model on the new data and update the target q net
        history = self.q_net.fit(X,Y,epochs = 10, verbose = 0, shuffle =True) 
        with open(self.save_location + "/loss.txt", 'a') as file: 
            for i in range(len(history.history["loss"])):
                file.write(str(history.history['loss'][i])+"\n")
            
    self.transitions = []

def reward_from_events(self, events):
    """
    *This is not a required function, but an idea to structure your code.*
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    move_penalty = -0.1
    game_rewards = {
        #Positive rewards
        #e.COIN_FOUND: 0.1, # encourages exploration
        e.COIN_COLLECTED: 5, 
        #e.CRATE_DESTROYED: 0.1,
        #e.KILLED_OPPONENT: 0.1,
        #e.SURVIVED_ROUND: 0.0, # encourages survival

        #Move penaltys
        e.MOVED_DOWN: move_penalty, # encourages efficent movement
        e.MOVED_LEFT: move_penalty,
        e.MOVED_RIGHT: move_penalty,
        e.MOVED_UP: move_penalty,
        e.BOMB_DROPPED: move_penalty,
        e.WAITED: -5, # waiting is as bad as doing something stupid
        
        #Stupid penaltys
        e.INVALID_ACTION: -5, # encourages to not be stupid [STRONG]
        #e.GOT_KILLED: 0,
        e.KILLED_SELF: -5
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum