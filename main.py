# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy.random import choice
import argparse
import os
from datetime import datetime
import math

from scipy.special import softmax

class AgentHistory:
    def __init__(self, id, T):

        self.id = id
        self.played_arms = np.zeros(T)
        self.observed_loss = np.zeros(T)
        self.communication_cost = np.zeros(T)
        self.agents_observed = []


def get_baselines(args):

    exp3 = math.sqrt(args.T * args.K)
    exp4 = 2.0 * math.sqrt(3) * math.sqrt(args.T * args.N * math.log(args.N, 2)) + 1
    fi = math.sqrt(args.T * math.log(args.K, 2))

    return exp3, exp4, fi


def get_learning_rate(args, communication_costs):

    c_tag = args.c # TODO: change this
    lr = 1.0 / (math.pow(c_tag/math.log(args.K, 2), (2.0/3.0)) + math.sqrt((args.T * args.K) / math.log(args.K, 2)))
    print("Learning rate {}".format(lr))
    return lr


def get_distributions(weights, lr):

    w = -lr * weights
    return softmax(w, axis=1)


def draw(arms_ids, probability_distribution, n_recs=1):

    arm = choice(arms_ids, size=n_recs, p=probability_distribution, replace=False)
    return arm


def update_weights(weights, loss, queried_agents, observed_arms, lr, communication_prob, arms_dist):

    # loss for the observed arms (played by agent and obtained by communication with others)
    a_t = observed_arms[np.where(queried_agents)]
    l_t = np.zeros(loss.shape)
    l_t[a_t] = loss[a_t]

    indicator_prob = np.zeros(loss.shape)
    indicator_prob[a_t] = 1.0 / 1.0 - np.prod((1.0 - communication_prob * arms_dist[:, a_t].T), axis=1)


    l_hat = np.divide(l_t, indicator_prob, out=np.zeros_like(l_t), where=indicator_prob != 0)
    return weights * np.exp(-lr * l_hat)


def choose_agent_to_query(id, communication_costs, arms_dist, lr):

    # personal_dist = arms_dist[id]
    p_m_t_i = np.sum(arms_dist, axis=0)
    # dest_div = np.sum(arms_dist / personal_dist, axis=1)
    # z probabilities
    # sample_prob = 1.0/arms_dist.shape[0] * np.sqrt( (lr/2) * dest_div / communication_costs[t])
    a_t = np.dot(arms_dist, 1/p_m_t_i)
    _const = (2.0 * (1.0 - np.exp(-1)))
    sample_prob = np.sqrt((lr /_const) * a_t * 1 /communication_costs)
    sample_prob[sample_prob > 1.0 ] = 1.0
    # include yourself, we will use it in weights updates
    sample_prob[id] = 1.0
    coin_flips = np.random.binomial(n=1, p=sample_prob)
    # Subtract personal cost
    tot_communication_cost = np.sum(communication_costs[np.where(coin_flips)]) - communication_costs[id]
    return coin_flips, tot_communication_cost, sample_prob


def initialize_history(T, N):
    h = []
    for i in range(N):
        h.append(AgentHistory(id=i, T=T))

    return h


def make_logdir(args, lr ):

    logdir = "./logs/T_{}_K_{}_N_{}_lr_{}_seed_{}_costs_{}/".format(args.T, args.K, args.N, lr, args.seed,
                                                                    args.c or 'random' ) + \
             datetime.now().strftime("%Y%m%d-%H%M%S")

    os.makedirs(logdir)

    return logdir

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--T', dest='T', type=int,  help='Horizontal time', default=100)
parser.add_argument('--K', dest='K', type=int,  help='Number of arms')
parser.add_argument('--cost', dest='c', type=float,  help='Number of arms', default=None)
parser.add_argument('--N', dest='N', type=int, help='Number of cooperating agents')
parser.add_argument('--lr', dest='lr', type=float,  help='learning rate', default=None)
parser.add_argument('--seed', dest='seed', type=int, default=42)
args = parser.parse_args()


def main(args):

    T = args.T
    K = args.K
    N = args.N
    lr = args.lr
    c = args.c


    np.random.seed(args.seed)
    # generate oblivious adversary losses
    adv_loss_map = np.random.random_sample((T, K))
    arms_ids = np.arange(K, dtype=np.int)
    # generate distribution of communication costs
    if c is not None:
        communication_costs = np.ones((T, N)) * c
        print("Using const costs")
    else:
        communication_costs = np.random.random_sample((T, N))
        print("Using rand costs")

    # initialize uniformly arm's distribution (row per agent)
    weights = np.ones((N, K))
    # arms_dist = np.ones((N, K)) * 1.0/K

    # initialize history
    observed_arms = np.zeros((T, N), dtype=np.int)
    observed_loss = np.zeros((T, N, 2))
    queried_agents = np.zeros((T, N, N), dtype=np.int)
    communication_prob = np.zeros((N, N))
    # history = initialize_history(T, N)

    if lr is None:
        lr = get_learning_rate(args, communication_costs)

    logdir = make_logdir(args, lr)
    min_loss = np.min(np.sum(adv_loss_map, axis=0))
    exp3, exp4, fi = get_baselines(args)
    print("Baselines EXP3:[{}] EXP4:[{}] FI:[{}]".format(exp3, exp4, fi))

    for t in tqdm(range(T)):

        loss_a_t = adv_loss_map[t]
        costs_t = communication_costs[t]
        arms_dist = get_distributions(weights, lr=lr)

        for agent in range(N):

            a_t = draw(arms_ids, arms_dist[agent])
            l_t = loss_a_t[a_t]
            observed_arms[t][agent] = a_t

            q_agents, costs, z = choose_agent_to_query(id=agent, communication_costs=communication_costs[t],
                                                            arms_dist=arms_dist, lr=lr)
            communication_prob[agent] = z
            # store indexes of agent that where queried for history purpose
            queried_agents[t][agent] = q_agents

            # store loss for played arm and communication cost (for history purpose)
            observed_loss[t][agent][0] = l_t
            observed_loss[t][agent][1] = costs

        # update weights
        for agent in range(N):
            weights[agent] = update_weights(weights=weights[agent], loss=loss_a_t,
                                            queried_agents=queried_agents[t][agent], observed_arms=observed_arms[t],
                                            lr=lr, communication_prob=communication_prob[agent], arms_dist=arms_dist)
        # # update probabilities
        # for agent in range(N):
        #
        #     arms_dist[t][N] = update_weights(weights=weights, observed_arms[t])

    for agent in range(N):
        obs_loss = np.sum(observed_loss[:, agent, 0])
        com_costs = np.sum(observed_loss[:, agent, 1])

        print("Agent [{}] Regret [{}] Played Loss [{}] Communication Loss [{}]".
              format(agent, obs_loss + com_costs - min_loss, obs_loss, com_costs))

if __name__ == '__main__':
    main(args)