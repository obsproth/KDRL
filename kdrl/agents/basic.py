import numpy as np
from .core import AbstractAgent
from _operator import index


class QLearningAgent(AbstractAgent):
    def __init__(self, action_space, state_shape, policy, *args, lr=1, gamma=0.99, Q_initializer='zeros', **kwargs):
        if not isinstance(action_space, int):
            raise NotImplementedError()
        super().__init__(action_space, *args, **kwargs)
        if isinstance(state_shape, int):
            state_shape = (state_shape,)
        else:
            state_shape = tuple(state_shape)
        self.state_shape = state_shape
        self.policy = policy
        self.lr = 1
        self.gamma = gamma
        from keras import backend as K
        from keras import initializers as I
        Q_shape = state_shape + (self.action_space,)
        self.Q = K.eval(I.get(Q_initializer)(Q_shape))
        #
        self.last_state = None
        self.last_action = None

    def start_episode(self, state):
        assert self.last_state is None
        assert self.last_action is None
        #
        action = self._select_action(state)
        self.last_state = state
        self.last_action = action
        return action

    def step(self, state, reward):
        assert self.last_state is not None
        assert self.last_action is not None
        #
        old_q = self.Q[self.last_state][self.last_action]
        new_q = reward + self.gamma * np.max(self.Q[state])
        self.Q[self.last_state][self.last_action] = (1 - self.lr) * old_q + self.lr * new_q
        #
        action = self._select_action(state)
        self.last_state = state
        self.last_action = action
        return action

    def end_episode(self, state, reward):
        old_q = self.Q[self.last_state][self.last_action]
        new_q = reward
        self.Q[self.last_state][self.last_action] = (1 - self.lr) * old_q + self.lr * new_q
        #
        self.last_state = None
        self.last_action = None

    def _select_action(self, state):
        scores = self.Q[tuple(state)]
        action = self.policy(scores)
        return action

    def select_best_action(self, state):
        return np.argmax(self.Q[tuple(state)])
