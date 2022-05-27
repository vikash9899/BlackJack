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


# monte Carlo Algorithm
def Monte_carlo(env, episodes, gamma=1., epsilon=0.2):
    sa_values = np.zeros((env.nS+1, env.nA), dtype=np.float16)
    visits = np.zeros((env.nS+1, env.nA), dtype=np.int16)
    Actions = ['Hit', 'Stick']

    epiosdes_history = []
    for i in range(episodes):
        # print("Episode ", i, " :- ")

        ob = env.start()
        state = ob['nState']
        Gt = 0
        ep_detail = []
        game_over = False
        while not game_over:
            # greedy policy improvement.
            prob = epsilon_greedy_policy(sa_values, state, epsilon, env.nA)

            # take the action
            action = np.random.choice(np.arange(len(prob)), p=prob)

            # Take the Action
            ob = env.play(Actions[action])

            ep_detail.append((state, action, ob['reward']))
            state = ob['nState']
            game_over = ob['done']

        epiosdes_history.append(ep_detail[-1][2])

        while len(ep_detail) != 0:
            ep_detail.reverse()

            S, A, R = ep_detail.pop()

            Gt = R + gamma * Gt

            visits[S][A] = visits[S][A] + 1

            sa_values[S][A] = sa_values[S][A] + \
                (Gt - sa_values[S][A] / visits[S][A])

    return sa_values, epiosdes_history
