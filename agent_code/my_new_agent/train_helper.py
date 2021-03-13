from .callbacks_helper import * 
import events as e


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
       
        current_q = self.q_net(old_state)
        target_q = np.copy(current_q)[0]
        next_q = self.target_q_net(new_state)
    
        # correct the prediction for highest rewards
        target_q[action] = reward + 0.7 * np.amax(next_q)
       
        #print(action,ACTIONS[action])
        #print('current'+str(np.array(current_q)))
        #print('target  '+ str(target_q))
        X.append(old_state)
        Y.append(target_q)
        
    '''old_state, action, reward, new_state = self.transitions[-1] # NOT ignore last move
    current_q = self.q_net(old_state)
    target_q = np.copy(current_q)[0]
    target_q[action] = reward
    
    X.append(old_state[0])
    Y.append([target_q])'''

    
    X = np.array(X)
    X = np.squeeze(X)
    Y = np.array(Y)

    # train the model on the new data and update the target q net
    history = self.q_net.fit(x = X,y = Y, verbose=0) 
    self.transitions = []


def reward_from_events(self, events):
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        #Positive rewards
        e.COIN_FOUND: .1, # encourages exploration
        e.COIN_COLLECTED: .5, 
        e.KILLED_OPPONENT: .1,
        e.SURVIVED_ROUND: 0, # encourages survival

        #Move penaltys
        e.MOVED_DOWN: .1, # encourages efficent movement
        e.MOVED_LEFT: .1,
        e.MOVED_RIGHT: .1,
        e.MOVED_UP: .1,
        e.WAITED: .1,
        
        #Stupid penaltys
        e.INVALID_ACTION: 0, # encourages to not be stupid
        e.GOT_KILLED: 0,
        e.KILLED_SELF: 0
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

