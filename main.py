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
import sys
from datetime import datetime
import math
from scipy.special import softmax


from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, log_dir):
        self.terminal = sys.stdout
        self.log = open(os.path.join(log_dir, "logfile.log"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


class AgentHistory:
    def __init__(self, id, T):

        self.id = id
        self.played_arms = np.zeros(T)
        self.observed_loss = np.zeros(T)
        self.communication_cost = np.zeros(T)
        self.agents_observed = []


def get_ub(args, t=None):

    if t is None:
        exp3 = math.sqrt(2 * args.T * args.K * math.log(args.K, 2))
        exp4 = 2.0 * math.sqrt(3) * math.sqrt(args.T * args.N * math.log(args.N, 2)) + 1
        fi = math.sqrt(args.T * math.log(args.K, 2))
    else:
        exp3 = math.sqrt(2 * t * args.K * math.log(args.K, 2))
        exp4 = 2.0 * math.sqrt(3) * math.sqrt(t * args.N * math.log(args.N, 2)) + 1
        fi = math.sqrt(t * math.log(args.K, 2))

    return exp3, exp4, fi


def get_learning_rate(args, communication_costs):

    if args.exp3:
        lr = np.sqrt(2.0 * math.log(args.K, 2) / (args.T * args.K))

    else:
        # c_tag = args.T * (args.N - 1) * np.sqrt(args.c)
        c_tag = np.sqrt(args.c)
        lr = 1.0 / (math.pow(c_tag/math.log(args.K, 2), (2.0/3.0)) +
                    math.sqrt((args.T * min(args.K, args.N)) / math.log(args.N, 2)))
        print("Learning rate {}".format(lr))

    return lr


def get_distributions(weights, lr):

    return weights / np.sum(weights, axis=1, keepdims=True)

    # w = -lr * weights
    # return softmax(w, axis=1)


def draw(arms_ids, probability_distribution, n_recs=1):

    arm = choice(arms_ids, size=n_recs, p=probability_distribution, replace=False)
    return arm


def update_weights(weights, loss, queried_agents, observed_arms, lr, communication_prob, arms_dist):

    # loss for the observed arms (played by agent and obtained by communication with others)
    a_t = observed_arms[np.where(queried_agents)]
    l_t = np.zeros(loss.shape)
    l_t[a_t] = loss[a_t]

    indicator_prob = np.zeros(loss.shape)
    indicator_prob[a_t] = 1.0 - np.prod((1.0 - communication_prob * arms_dist[:, a_t].T), axis=1)

    l_hat = np.divide(l_t, indicator_prob, out=np.zeros_like(l_t), where=indicator_prob != 0)
    return weights * np.exp(-lr * l_hat)


def choose_agent_to_query(id, communication_costs, arms_dist, lr, exp3):

    # personal_dist = arms_dist[id]
    p_m_t_i = np.sum(arms_dist, axis=0)
    # dest_div = np.sum(arms_dist / personal_dist, axis=1)
    # z probabilities
    # sample_prob = 1.0/arms_dist.shape[0] * np.sqrt( (lr/2) * dest_div / communication_costs[t])
    a_t = np.dot(arms_dist, 1/p_m_t_i)
    _const = (2.0 * (1.0 - np.exp(-1)))

    # if a cost is zero we sample all agents
    if np.all(np.array(communication_costs) == 0):
        sample_prob = np.ones_like(a_t)

    else:
        sample_prob = np.sqrt((lr / _const) * a_t * 1 / communication_costs)

    sample_prob[sample_prob > 1.0] = 1.0

    # do not sample other agents
    if exp3:
        sample_prob = np.zeros_like(sample_prob)

    # include yourself, we will use it in weights updates
    sample_prob[id] = 1.0
    coin_flips = np.random.binomial(n=1, p=sample_prob)
    # Subtract personal cost
    tot_communication_cost = np.sum(communication_costs[np.where(coin_flips)]) - communication_costs[id]
    return coin_flips, tot_communication_cost, sample_prob


def choose_agent_to_query_by_KS(id, communication_costs, arms_dist, lr, exp3):

    def suprimum_dist(x, y):
        return max(np.abs(x - y))

    personal_dist = arms_dist[id]
    distances = np.apply_along_axis(suprimum_dist, 1, arms_dist, personal_dist)

    # z probabilities
    sample_prob = np.sqrt(arms_dist.shape[1]/arms_dist.shape[0]) * np.sqrt((lr/2)) * distances
    # if a cost is zero we sample all agents
    if np.all(np.array(communication_costs) == 0):
        sample_prob = np.ones_like(sample_prob)

    else:
        sample_prob = sample_prob * (1 / np.sqrt(communication_costs))

    sample_prob[sample_prob > 1.0] = 1.0

    # do not sample other agents
    if exp3:
        sample_prob = np.zeros_like(sample_prob)

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

    c = 'random' if args.c is None else args.c

    log_dir = "T_{}_K_{}_N_{}_lr_{}_ls_{}_seed_{}_costs_{}_z_{}/".format(args.T, args.K, args.N, lr, args.ls, args.seed,
                                                                    c, args.z) + \
             datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.exp3:
        log_dir = "Exp3_" + log_dir

    log_dir = './logs/' + log_dir
    os.makedirs(log_dir)
    return log_dir


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--T', dest='T', type=int,  help='Horizontal time', default=100)
parser.add_argument('--K', dest='K', type=int,  help='Number of arms')
parser.add_argument('--cost', dest='c', type=float,  help='Number of arms', default=None)
parser.add_argument('--N', dest='N', type=int, help='Number of cooperating agents')
parser.add_argument('--lr', dest='lr', type=float,  help='learning rate', default=None)
parser.add_argument('--ls', dest='ls', type=float,  help='best arm loss significance', default=0.5)
parser.add_argument('--seed', dest='seed', type=int, default=42)
parser.add_argument('--exp3', dest='exp3', action='store_true', default=False)
parser.add_argument('--z', dest='z', type=str,  help='z alg', default='KS')

args = parser.parse_args()


def main(args):

    T = args.T
    K = args.K
    N = args.N
    lr = args.lr
    c = args.c

    # generate distribution of communication costs
    if c is not None:
        communication_costs = np.ones((T, N)) * c
        print("Using const costs")
    else:
        communication_costs = np.random.random_sample((T, N))
        print("Using rand costs")

    if lr is None:
        lr = get_learning_rate(args, communication_costs)

    log_dir = make_logdir(args, lr)
    sys.stdout = Logger(log_dir=log_dir)

    if args.exp3:
        print("RUNNING EXP3...")

    np.random.seed(args.seed)
    # generate oblivious adversary losses
    adv_loss_map = np.random.random_sample((T, K))
    arm_idx = np.random.choice(K)
    adv_loss_map[np.random.choice(T, int(args.ls * T)), arm_idx:arm_idx + 1] = 0

    mean_loss = np.mean(np.sum(adv_loss_map, axis=0))
    min_loss = np.min(np.sum(adv_loss_map, axis=0))
    best_arm_id = np.argmin(np.sum(adv_loss_map, axis=0))
    # sanity check
    assert arm_idx == best_arm_id
    print("Mean Loss: [{}]\tBest Arm [{}] with total loss [{}]".format(mean_loss, best_arm_id, min_loss))

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
    writer = SummaryWriter(log_dir=log_dir)
    exp3, exp4, fi = get_ub(args)
    print("Upper Bound EXP3:[{}] EXP4:[{}] FI:[{}]".format(exp3, exp4, fi))

    for t in tqdm(range(T)):

        exp3, exp4, fi = get_ub(args, t)
        loss_a_t = adv_loss_map[t]
        costs_t = communication_costs[t]
        arms_dist = get_distributions(weights, lr=lr)

        for agent in range(N):

            a_t = draw(arms_ids, arms_dist[agent])
            l_t = loss_a_t[a_t]
            observed_arms[t][agent] = a_t

            if args.z == 'KS':
                q_agents, costs, z = choose_agent_to_query_by_KS(id=agent, communication_costs=communication_costs[t],
                                                           arms_dist=arms_dist, lr=lr, exp3=args.exp3)
            else:
                q_agents, costs, z = choose_agent_to_query(id=agent, communication_costs=communication_costs[t],
                                                                arms_dist=arms_dist, lr=lr, exp3=args.exp3)

            communication_prob[agent] = z
            # store indexes of agent that where queried for history purpose
            queried_agents[t][agent] = q_agents
            if t % 5000 == 0:
                z_stat = np.delete(z, [agent])
                print("Round [{}] Agent [{}]".format(t, agent))
                # print("Z probabilities {}".format(z))
                print('Z mean: [{}] , Z std [{}]'.format(np.mean(z_stat), np.std(z_stat)))

            # store loss for played arm and communication cost (for history purpose)
            observed_loss[t][agent][0] = l_t
            observed_loss[t][agent][1] = costs

        best_arm_loss = np.sum(adv_loss_map[:t+1, best_arm_id])
        avg_com_loss = np.sum(observed_loss[:t+1,:,1]) / N
        avg_obs_loss = np.sum(observed_loss[:t+1,:,0]) / N

        writer.add_scalars('Accumulated Loss', {'best_arm': best_arm_loss,
                                       'Avg Com Cost': avg_com_loss,
                                       'Avg Observed Cost': avg_obs_loss, }, t)

        writer.add_scalars('Regret', {'exp3_UB': exp3,
                                       'exp4_UB': exp4,
                                      'fi_UB': fi,
                                       'regret': avg_com_loss + avg_obs_loss - best_arm_loss, }, t)

        writer.add_histogram('sampled_agents',  queried_agents[t], t)
        writer.add_histogram('played_arms', observed_arms[t], t)

        # update weights
        for agent in range(N):
            weights[agent] = update_weights(weights=weights[agent], loss=loss_a_t,
                                            queried_agents=queried_agents[t][agent], observed_arms=observed_arms[t],
                                            lr=lr, communication_prob=communication_prob[agent], arms_dist=arms_dist)
        # # update probabilities
        # for agent in range(N):
        #
        #     arms_dist[t][N] = update_weights(weights=weights, observed_arms[t])

    print("Best Arm [{}]".format(best_arm_id))
    regrets = np.zeros(N)

    for agent in range(N):
        obs_loss = np.sum(observed_loss[:, agent, 0])
        com_costs = np.sum(observed_loss[:, agent, 1])

        regrets[agent] = (obs_loss + com_costs - min_loss)
        print("Agent [{}] Regret [{}] Played Loss [{}] Communication Loss [{}]".
              format(agent, regrets[agent], obs_loss, com_costs))

    print("Agents mean regret [{}] mean played loss [{}] mean comm loss [{}] ".format(np.mean(regrets),
                                                                           np.sum(observed_loss[:, :, 0])/N,
                                                                           np.sum(observed_loss[:, :, 1])/N))


if __name__ == '__main__':
    main(args)