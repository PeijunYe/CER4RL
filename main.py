import json
import random
import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal

from PPO import Agent, get_args

from carla_env.env import CarEnv

env = CarEnv()
cfg = get_args()
agent = Agent(cfg)
max_seq_counter = 1000
max_seq_length = 1000
turn = 100


def totals(data):
    all_reward_chains_per = []
    for j in range(len(data)):
        total = 0
        for i in range(len(data[j])):
            total = total + data[j][i]

        if total == 5 and len(data[j]) < 30:
            total = 0

        if total == -1 and len(data[j]) > 20:
            total = 4

        if total == 0:
            total = 0.001 * len(data[j])

        all_reward_chains_per.append(total)
    return all_reward_chains_per


def select(mutate_chromosome_states, mutate_s_a_chains_reward, mutate_val, mutate_index, mutate_probs,
           chromosome_states,
           s_a_chains_reward, chromosome_val, chromosome_indexes, chromosome_probs,
           top_k):
    chromosome_states_combination = chromosome_states + mutate_chromosome_states
    pop_total_reward_list = totals(s_a_chains_reward)
    mutate_pop_total_reward_list = totals(mutate_s_a_chains_reward)

    pop_total_reward_list_combination = pop_total_reward_list + mutate_pop_total_reward_list
    s_a_chains_reward_combination = s_a_chains_reward + mutate_s_a_chains_reward
    val_combination = chromosome_val + mutate_val
    index_combination = chromosome_indexes + mutate_index
    probs_combination = chromosome_probs + mutate_probs

    index = np.argsort(pop_total_reward_list_combination)[::-1][:top_k].tolist()

    chromosome_states_new = [chromosome_states_combination[i] for i in index]
    s_a_chains_reward_new = [s_a_chains_reward_combination[i] for i in index]
    val_new = [val_combination[i] for i in index]
    index_new = [index_combination[i] for i in index]
    probs_new = [probs_combination[i] for i in index]

    return chromosome_states_new, s_a_chains_reward_new, val_new, index_new, probs_new


def mutate_policy_sampling(state, action, cand_num):
    action_old = torch.as_tensor(action)
    action_all = []
    distance = []
    oushi_distance = nn.PairwiseDistance(p=2)
    for i in range(100):
        action, prob, val, action_index = agent.choose_action(state)
        action = [torch.squeeze(action[0]).item()] + [torch.squeeze(action[1]).item()]
        action = torch.tensor(action)
        output = oushi_distance(action_old, action)
        distance.append(output)
        action_all.append(action)

    distance = torch.tensor(distance)
    x = torch.topk(distance, cand_num)
    normal_action_index = x.indices[0:cand_num]
    action_list = [action_all[i] for i in normal_action_index]
    action_from_multinormal = collect_samples(action_all, cand_num=50)
    action_total = action_list + action_from_multinormal
    action = random.sample(action_total, 1)
    return action


def collect_samples(action_all, cand_num):
    a = action_all[0]
    for ele in range(0, len(action_all) - 1):
        b = action_all[ele + 1]
        a = torch.cat((a, b))
    mean_terminal = torch.mean(a.reshape(len(action_all), 2), dim=0)
    cov = torch.cov(a.reshape(len(action_all), 2).T)
    m = MultivariateNormal(mean_terminal, cov)
    action_from_multinormal = []
    probs = []
    while len(probs) < cand_num:
        x = m.sample()
        p = m.log_prob(x)
        prob = torch.exp(p)
        if prob < 0.1:
            action_from_multinormal.append(x)
            probs.append(prob)
    return action_from_multinormal


def init():
    s_a_seqes = []
    rewards = []
    init_probs = []
    init_vals = []
    init_action_indexes = []
    for i in range(max_seq_counter):
        s_a_seq = []
        chain_rewrad = []
        init_prob = []
        init_val = []
        init_action_index = []
        state = env.reset()
        s_a_seq.append(state)

        for i in range(max_seq_length):
            action, prob, val, action_index = agent.choose_action(state)
            action = [torch.squeeze(action[0]).item()] + [torch.squeeze(action[1]).item()]
            s_a_seq.append(action)
            init_prob.append(prob)
            init_val.append(val)
            init_action_index.append(action_index)
            next_state, reward, done = env.step(action)
            chain_rewrad.append(reward)
            s_a_seq.append(next_state)
            if done:
                break

            state = next_state

        rewards.append(chain_rewrad)
        s_a_seqes.append(s_a_seq)
        init_probs.append(init_prob)
        init_vals.append(init_val)
        init_action_indexes.append(init_action_index)

    return s_a_seqes, rewards, init_probs, init_vals, init_action_indexes


def rebuild_fuction(chromosome_states, init_rewards, init_probs, init_vals, init_action_indexes):
    mutate_rebuild_states = []
    mutate_rebuild_s_a_chains_reward = []
    mutate_rebuild_val_lists = []
    mutate_rebuild_action_index_lists = []
    mutate_rebuild_prob_lists = []

    for mutate_person in range(len(chromosome_states)):
        mutate_rebuild_state_average_reward_list = []
        mutate_rebuild_val_list = []
        mutate_rebuild_action_index_list = []
        mutate_rebuild_prob_list = []
        rebuild_state = []
        t = random.randint(0, len(init_rewards[mutate_person]) - 1)
        mutate_action_index = 2 * t + 1

        action = mutate_policy_sampling(chromosome_states[mutate_person][mutate_action_index - 1],
                                        chromosome_states[mutate_person][mutate_action_index],
                                        cand_num=50)

        action = action[0].tolist()
        remain = chromosome_states[mutate_person][:mutate_action_index]
        reamin_reward_index = t
        remain_reward_list = init_rewards[mutate_person][:reamin_reward_index]
        remain_val_list = init_vals[mutate_person][:reamin_reward_index]
        remain_probs_list = init_probs[mutate_person][:reamin_reward_index]
        remain_action_index_list = init_action_indexes[mutate_person][:reamin_reward_index]

        for n in range(len(remain_reward_list)):
            mutate_rebuild_state_average_reward_list.append(remain_reward_list[n])
            mutate_rebuild_val_list.append(remain_val_list[n])
            mutate_rebuild_action_index_list.append(remain_action_index_list[n])
            mutate_rebuild_prob_list.append(remain_probs_list[n])

        for j in range(len(remain)):
            rebuild_state.append(remain[j])

        next_state, reward, done = env.step(action)
        rebuild_state.append(action)
        mutate_rebuild_val_list.append(0.0)
        mutate_rebuild_action_index_list.append(0)
        mutate_rebuild_prob_list.append(0.0)

        rebuild_state.append(next_state[1:].tolist())

        mutate_rebuild_state_average_reward_list.append(reward)
        if done:
            mutate_rebuild_states.append(rebuild_state)
            mutate_rebuild_s_a_chains_reward.append(mutate_rebuild_state_average_reward_list)
            mutate_rebuild_val_lists.append(mutate_rebuild_val_list)
            mutate_rebuild_action_index_lists.append(mutate_rebuild_action_index_list)
            mutate_rebuild_prob_lists.append(mutate_rebuild_prob_list)
            continue

        next_state_index = mutate_action_index + 1
        state = next_state[1:].tolist()

        for k in range(((max_seq_length * 2 + 1) - next_state_index) // 2):
            action, prob, val, action_index = agent.choose_action(state)
            action = [torch.squeeze(action[0]).item()] + [torch.squeeze(action[1]).item()]
            rebuild_state.append(action)
            mutate_rebuild_val_list.append(val)
            mutate_rebuild_action_index_list.append(action_index)
            mutate_rebuild_prob_list.append(prob)
            next_state, reward, done = env.step(action)
            mutate_rebuild_state_average_reward_list.append(reward)
            rebuild_state.append(next_state[1:].tolist())
            if done:
                break
            state = next_state[1:].tolist()

        mutate_rebuild_states.append(rebuild_state)
        mutate_rebuild_s_a_chains_reward.append(mutate_rebuild_state_average_reward_list)
        mutate_rebuild_val_lists.append(mutate_rebuild_val_list)
        mutate_rebuild_action_index_lists.append(mutate_rebuild_action_index_list)
        mutate_rebuild_prob_lists.append(mutate_rebuild_prob_list)

    return mutate_rebuild_states, mutate_rebuild_s_a_chains_reward, mutate_rebuild_val_lists, mutate_rebuild_action_index_lists, mutate_rebuild_prob_lists


def calculate_reward(rewards):
    total_reward = []
    for i in range(len(rewards)):
        chain_total_reward = 0
        for j in range(len(rewards[i])):
            chain_total_reward = chain_total_reward + rewards[i][j]
        total_reward.append(chain_total_reward)
    res = sum(total_reward)

    return res


def collect_data(chromosome_states, action_indexes, probs, vals, rewrads):
    steps = 0
    for i in range(len(rewrads)):
        for j in range(len(rewrads[i])):
            chromosome_state_new = chromosome_states[i][2 * j]
            chromosome_state_action_index = action_indexes[i][j]
            chromosome_state_action_probs = probs[i][j]
            chromosome_state_action_vals = vals[i][j]
            chromosome_state_rewards = rewrads[i][j]
            if j == len(rewrads[i]):
                done = True
            else:
                done = False
            agent.memory.push(chromosome_state_new, chromosome_state_action_index, chromosome_state_action_probs,
                              chromosome_state_action_vals, chromosome_state_rewards, done)

            steps += 1
        agent.learn()


def list_2_numpy(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if j % 2 == 0:
                data[i][j] = np.array(data[i][j])
    return data


def init_from_file(chromo_filename, reward_filename, val_file, action_index_file, probs_file):
    with open(chromo_filename) as file_obj:
        init_chromosome_states = json.load(file_obj)

    with open(reward_filename) as file_obj:
        init_s_a_chains_reward = json.load(file_obj)

    with open(val_file) as file_obj:
        init_val = json.load(file_obj)

    with open(action_index_file) as file_obj:
        init_action_indexes = json.load(file_obj)

    with open(probs_file) as file_obj:
        init_probs = json.load(file_obj)

    chromosome_states = init_chromosome_states
    s_a_chains_reward = init_s_a_chains_reward

    return chromosome_states, s_a_chains_reward, init_val, init_action_indexes, init_probs


def train():
    init_s_a_seqes, init_rewards, init_probs, init_vals, init_action_indexes = init()
    for i in range(turn):
        rebuild_s_a_seqes, rebuild_mutate_s_a_chains_reward, rebuild_val_lists, rebuild_index_lists, rebuid_prob_lists = rebuild_fuction(
            init_s_a_seqes,
            init_rewards, init_probs,
            init_vals, init_action_indexes)

        s_a_seqes_new, rewards_new, val_new, index_new, prob_new = select(rebuild_s_a_seqes,
                                                                          rebuild_mutate_s_a_chains_reward,
                                                                          rebuild_val_lists,
                                                                          rebuild_index_lists,
                                                                          rebuid_prob_lists,
                                                                          init_s_a_seqes, init_rewards,
                                                                          init_vals,
                                                                          init_action_indexes,
                                                                          init_probs, len(init_rewards))

        collect_data(s_a_seqes_new, index_new, prob_new, val_new, rewards_new)

        init_chromosome_states = s_a_seqes_new
        init_rewards = rewards_new
        init_probs = prob_new
        init_vals = val_new
        init_action_indexes = index_new

        if turn % 10 == 0:
            with open(chromo_file, 'w') as file_obj:
                json.dump(init_chromosome_states, file_obj)

            with open(reward_file, 'w') as file_obj:
                json.dump(init_rewards, file_obj)

            with open(val_file, 'w') as file_obj:
                json.dump(init_vals, file_obj)

            with open(action_index_file, 'w') as file_obj:
                json.dump(init_action_indexes, file_obj)

            with open(probs_file, 'w') as file_obj:
                json.dump(init_probs, file_obj)

            print(' iteration saved...')

    torch.save(agent.actor.state_dict(), 'actor_params.pth')

    torch.save(agent.critic.state_dict(), 'critic_params.pth')


def eval():
    actor = Actor(n_states, n_actions, cfg.hidden_dim)
    state_dict_actor = torch.load('actor_params.pth')
    actor.load_state_dict(state_dict_actor)

    critic = Critic(n_states, cfg.hidden_dim)
    state_dict_critic = torch.load('critic_params.pth')
    critic.load_state_dict(state_dict_critic)
    for iter in range(5):
        s_a_seqes, rewards, init_probs, init_vals, init_action_indexes = init()

        total_chain_reward = (calculate_reward(rewards))

        print('iter {}\t  eval_R {} \t '.format(iter + 1, total_chain_reward))


if __name__ == '__main__':
    train()
    # eval()
