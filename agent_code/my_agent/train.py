import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import reshape_game_state

PLACEHOLDER_EVENT = 'INVALID_ACTION'

def do_training(self, batch):
    '''
    take a batch of experiences and update the model
    '''
    old_game_state_batch, action_batch, reward_batch, new_game_state_batch = batch

    current_q = self.q_net(old_state_batch)
    target_q = np.copy(current_q)
    next_q = self.target_q_net(next_state_batch)
    max_next_q = np.amax(next_q, axis=1)
    for i in range(old_game_state_batch.shape[0]):
        target_q[i][action_batch[i]] = reward_batch[i] + 0.95 * max_next_q[i]
    result = self.q_net.fit(x=state_batch, y=target_q)

@add_method(self)
def get_gameplay_batch(size):
    return self.transition[size-2:-2]

#Moritz: Initialize the NN here?
def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.do_training = do_training
    self.reward_from_events_TEST = reward_from_events_TEST
    self.transitions = []
    self.get_gameplay_batch = get_gameplay_batch


#Severin: Collect rewards and fill experience buffer
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # state_to_features is defined in callbacks.py
    if new_game_state['round']==1:
        old_game_state = None
        reward = 0
        action = None
    else:
        old_game_state = self.reshape_game_state(old_game_state)
        reward = self.reward_from_events_TEST(events)
    new_game_state = self.reshape_game_state(new_game_state)

    self.transitions.append([old_game_state,action,reward,new_game_state])


#Severin: Step2 --> Check results and trigger gradient descent
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    last_game_state = self.reshape_game_state(last_game_state)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(events)
    reward = self.reward_from_events_TEST(events)

    self.transitions.append([last_game_state,last_action,reward,None])

    '''# Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
    '''
    batch = self.get_gameplay_batch(4)
    self.train(batch)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def reward_from_events_TEST(events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum