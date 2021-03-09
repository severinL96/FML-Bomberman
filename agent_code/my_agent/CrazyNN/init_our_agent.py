  def build_q_network(n_actions,input_shape, learning_rate=0.00001):
    """Builds a DQN as a Keras model
    
    Arguments:
        n_actions: Number of possible action the agent can take
        learning_rate: Learning rate
        input_shape: Shape of the preprocessed frame the model sees
        history_length: Number of historical frames the agent can see
        
    Returns:
        A compiled Keras model
    """
    model_input = Input(shape=(input_shape))

    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    
    q_vals = Dense(n_actions)(x)  #activation ='softmax')(x)

    # Build model
    model = Model(model_input, q_vals)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model

#--------------------------------------------

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    '''
    sets up inital q_net and target_q_net
    '''
    self.q_net = build_q_network(n_actions=6,input_shape=1445, learning_rate=0.00001)
    self.target_q_net = build_q_network(n_actions=6,input_shape=1445, learning_rate=0.00001)

def act(self,state,ACTIONS,epsilon):
    '''
    Query the Q-Net for an action given a state (can be random action)
    Arguments: 
        state: state tp give action for
        ACTIONS: possible actions
        epsilon: The probability of doing a random move
    Returns:
        a string returning the predicted move
    '''
    
    if np.random.rand()<epsilon:
        return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
    else:
        state_input = self.state_to_features(state)
        action_q = self.q_net(state_input)
        action_index = np.argmax(action_q.numpy()[0], axis=0)
        return ACTIONS[action_index]

def update_target_q_net(self)
    '''update the target Q network'''
    self.target_q_net.set_weights(self.q_net.get_weights()) 

def predict_target(self,state)
    '''returns the estimated reward for a given state'''
    state_input = self.state_to_features(state)
    action_q = self.q_net(state_input)
    return np.argmax(action_q.numpy()[0], axis=0) 
