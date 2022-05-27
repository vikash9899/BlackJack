import matplotlib.pylab as plt
import seaborn as sns
from env import black_jack as bj
import numpy as np
import pandas as pd


# epsilon greedy policy
def epsilon_greedy_policy(sa_values, state, epsilon, nA):
    prob = np.ones(nA) * epsilon / nA
    prob[np.argmax(sa_values[state])] += 1 - epsilon
    # print(np.sum(prob))
    return prob


# q_learning Algorithm
def q_learning(env, episods, gamma=1, alpha=0.1, epsilon=0.3):

    sa_values = np.zeros((env.nS+1, env.nA), dtype=np.float16)
    visits = np.zeros((env.nS+1, env.nA), dtype=np.int16)
    Actions = ['Hit', 'Stick']

    for i_episod in range(1, episods+1):

        ob = env.start()
        state = ob['nState']
        game_over = False

        while not game_over:
            # greedy policy improvement.
            prob = epsilon_greedy_policy(sa_values, state, epsilon, env.nA)

            # take the action
            action = np.random.choice(np.arange(len(prob)), p=prob)

            # Take the Action
            ob = env.play(Actions[action])

            # ep_detail.append((state, action, ob['reward']))
            n_state = ob['nState']
            reward = ob['reward']
            game_over = ob['done']

            if not game_over:
                sa_values[state][action] += alpha * \
                    (reward + gamma *
                     np.max(sa_values[n_state]) - sa_values[state][action])
                state = n_state
            else:
                sa_values[state][action] += alpha * \
                    (reward + 0 - sa_values[state][action])

    return sa_values
