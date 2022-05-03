import matplotlib.pylab as plt
import seaborn as sns
from env import black_jack as bj
import numpy as np
import pandas as pd

'''
env = bj.BlackJack()

# print(env.nStates)
# print(env.nAction)
# print(env.Actions)
# print(env.States)
# print(env.nCards)
# print(env.dealers_cards)
# print(env.players_cards)
# print(env.dealers_hand)
# print(env.players_hand)

ob = env.play('Hit')
print("ob", ob)

ob1 = ()
if len(ob) > 1 and ob[5] == False:
    ob1 = env.play('Hit')

print('ob1', ob1)

ob2 = ()
if len(ob1) > 1 and ob1[5] == False: 
    ob2 = env.play('Stick') 

print('ob2', ob2)


env1 = bj.BlackJack() 
ob1 = env1.play('Stick') 
print(ob1) 
'''


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


# Sarsa Algorithm
def sarsa(env, episods, gamma=1, alpha=0.1, epsilon=0.3):

    sa_values = np.zeros((env.nS+1, env.nA), dtype=np.float16)
    visits = np.zeros((env.nS+1, env.nA), dtype=np.int16)
    Actions = ['Hit', 'Stick']

    for i_episod in range(1, episods+1):
        ob = env.start()
        state = ob['nState']
        game_over = False

        while not game_over:
            # greedy policy improvement.
            prob = epsilon_greedy_policy(sa_values, state, 0.2, env.nA)

            # take the action
            action = np.random.choice(np.arange(len(prob)), p=prob)

            # Take the Action
            ob = env.play(Actions[action])

            # ep_detail.append((state, action, ob['reward']))
            n_state = ob['nState']
            reward = ob['reward']
            game_over = ob['done']

            if not game_over:
                prob = epsilon_greedy_policy(sa_values, state, epsilon, env.nA)
                n_action = np.random.choice(np.arange(len(prob)), p=prob)
                sa_values[state][action] += alpha * \
                    (reward + gamma * sa_values[n_state]
                     [n_action] - sa_values[state][action])
                state = n_state
                action = n_action
            else:
                sa_values[state][action] += alpha * \
                    (reward + gamma * 0 - sa_values[state][action])

    return sa_values


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


def main():
    env = bj.BlackJack()
    Q, history = Monte_carlo(env, 10000)

    df = pd.DataFrame(Q)
    df.columns = ['Hit', 'Stick']
    df = df.loc[list(range(4, 22))]

    # plt.figure(figsize=(6, 10), dpi=100)
    plt.title("Monte carlo learning")
    # plt.xlabel("Actions ----- ")
    # plt.ylabel("States ----- ")
    s = sns.heatmap(df, linewidth=0.5, annot=True)
    s.set(xlabel='Actions', ylabel='States')

    plt.show()

    Q_sarsa = sarsa(env, 100000)
    print(Q_sarsa)

    df = pd.DataFrame(Q_sarsa)
    df.columns = ['Hit', 'Stick']
    df = df.loc[list(range(4, 22))]

    plt.title("Temporal difference learning ")
    s = sns.heatmap(df, linewidth=0.5, annot=True)
    s.set(xlabel='Actions', ylabel='States')

    plt.show()

    q_learn = q_learning(env, 100000)
    print(q_learn)

    df = pd.DataFrame(q_learn)
    df.columns = ['Hit', 'Stick']
    df = df.loc[list(range(4, 22))]

    plt.title("Q Learning")
    plt.xlabel("Actions ----- ")
    plt.ylabel("States ----- ")
    s = sns.heatmap(df, linewidth=0.5, annot=True)
    s.set(xlabel='Actions', ylabel='States')
    plt.show()


if __name__ == '__main__':
    main()
