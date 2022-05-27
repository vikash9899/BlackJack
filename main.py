from env import black_jack as bj
from agents import monet_carlo
from agents import q_learning
from agents import sarsa
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_Q_values(Q, title):
    df = pd.DataFrame(Q)
    df.columns = ['Hit', 'Stick']
    df = df.loc[list(range(4, 22))]

    plt.title(title)
    s = sns.heatmap(df, linewidth=0.5, annot=True)
    s.set(xlabel='Actions', ylabel='States')

    plt.show()


def main():
    env = bj.BlackJack()
    Q, history = monet_carlo.Monte_carlo(env, 10000)

    plot_Q_values(Q, "Monte Carlo Control")

    Q_sarsa = sarsa(env, 100000)
    plot_Q_values(Q_sarsa, "Temporal Difference Learning")

    q_learn = q_learning(env, 100000)
    plot_Q_values(q_learn, "Q learning")


if __name__ == '__main__':
    main()
