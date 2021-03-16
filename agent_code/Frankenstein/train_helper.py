from .callbacks_helper import * 
import events as e


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

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


def do_training(self):
    '''
    INPUT: Transitions

    Takes the transitions and calculates q values for the respective
    states. Trains the model to correctly predict an actions q value
    '''
    X = []
    Y = []
    
    for transition in self.transitions[1:-1]: # ingore first mov
        X.append(x1)
        X.append(x2)
        Y.append(y1)
        Y.append(y2)

    X = np.array(X)
    #X = np.squeeze(X)
    Y = np.array(Y)

    # train the model on the new data and update the target q net
    history = self.q_net.fit(x = X,y = Y, verbose=0) 
    with open(self.save_location + "/loss.txt", 'a') as file: 
            file.write(str(history.history['loss'][-1])+"\n")
    self.transitions = []

def do_training_with_PER_2(self):
    '''
    INPUT: Transitions

    Takes the transitions and calculates q values for the respective
    states. Trains the model to correctly predict an actions q value
    '''
    X = []
    Y = []    
    diff = [] 
    diff.append(-1000) # add spacer since we ignore first state
    for transition in self.transitions[1:-1]: # ingore first move
        old_state, action, reward, new_state = transition

        current_q = self.q_net(np.expand_dims(old_state,axis=0))
        target_q = np.copy(current_q)[0]
        next_q = self.target_q_net(np.expand_dims(new_state,axis=0))

    
        # correct the prediction for highest rewards
        target_q[action] = reward + 0.0 * np.amax(next_q)
        
        diff.append(np.linalg.norm(current_q - target_q))
       

    number_samples = min(len(diff)//2,10)
    indices = list(np.argpartition(diff,-number_samples)[-number_samples:])
    for transition in [self.transitions[i] for i in indices]:
        old_state, action, reward, new_state = transition

        current_q = self.q_net(np.expand_dims(old_state,axis=0))
        target_q = np.copy(current_q)[0]
        next_q = self.target_q_net(np.expand_dims(new_state,axis=0))

    
        # correct the prediction for highest rewards
        target_q[action] = reward + 0 * np.amax(next_q)
        
        diff.append(np.linalg.norm(current_q - target_q))
        self.logger.debug('action: '+str(action) +' ('+ ACTIONS[action]+') got reward: '+str(reward))
        self.logger.debug('current'+str(np.array(current_q)))
        self.logger.debug('target  '+ str(target_q))
  
        X.append(old_state)
        Y.append(target_q)
    X = np.array(X)
    #X = np.squeeze(X)
    Y = np.array(Y)

    # train the model on the new data and update the target q net
    history = self.q_net.fit(x = X,y = Y, verbose=0,batch_size = 16) 
    with open(self.save_location + "/loss.txt", 'a') as file: 
            file.write(str(history.history['loss'][0])+"\n")
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
        e.COIN_FOUND: 0.1, # encourages exploration
        e.COIN_COLLECTED: 5, 
        e.CRATE_DESTROYED: 0.1,
        e.KILLED_OPPONENT: 0.1,
        e.SURVIVED_ROUND: 0.0, # encourages survival

        #Move penaltys
        e.MOVED_DOWN: move_penalty, # encourages efficent movement
        e.MOVED_LEFT: move_penalty,
        e.MOVED_RIGHT: move_penalty,
        e.MOVED_UP: move_penalty,
        e.BOMB_DROPPED: move_penalty,
        e.WAITED: -5, # waiting is as bad as doing something stupid
        
        #Stupid penaltys
        e.INVALID_ACTION: -5, # encourages to not be stupid [STRONG]
        e.GOT_KILLED: 0,
        e.KILLED_SELF: -5
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

