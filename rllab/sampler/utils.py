import numpy as np
from rllab.misc import tensor_utils


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        if isinstance(r, (list, np.ndarray)):
            assert (r == r[0]).all()
            r = r[0]
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
    return dict(observations=tensor_utils.stack_tensor_list(observations),
                actions=tensor_utils.stack_tensor_list(actions),
                rewards=tensor_utils.stack_tensor_list(rewards),
                agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
                env_infos=tensor_utils.stack_tensor_dict_list(env_infos),)


def decrollout(env, agent, max_path_length=np.inf, animated=False, speedup=1):
    n_agents = len(env.agents)
    observations = [[] for _ in xrange(n_agents)]
    actions = [[] for _ in xrange(n_agents)]
    rewards = [[] for _ in xrange(n_agents)]
    agent_infos = [[] for _ in xrange(n_agents)]
    env_infos = [[] for _ in xrange(n_agents)]
    olist = env.reset()
    agent.reset()
    path_length = 0
    while path_length < max_path_length:
        alist = []
        for i, o in enumerate(olist):
            a, agent_info = agent.get_action(o)
            alist.append(a)
            observations[i].append(env.observation_space.flatten(o))
            actions[i].append(env.action_space.flatten(a))
            agent_infos[i].append(agent_info)

        next_olist, rlist, d, env_info = env.step(np.asarray(alist))
        for i, r in enumerate(rlist):
            rewards[i].append(r)
            env_infos[i].append(env_info)
        path_length += 1
        if d:
            break
        olist = next_olist
    trajs = [dict(observations=tensor_utils.stack_tensor_list(observations[i]),
                  actions=tensor_utils.stack_tensor_list(actions[i]),
                  rewards=tensor_utils.stack_tensor_list(rewards[i]),
                  agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos[i]),
                  env_infos=tensor_utils.stack_tensor_dict_list(env_infos[i]),)
             for i in xrange(n_agents)]
    return trajs


def chunk_decrollout(env, agent, max_path_length=np.inf, chunked_path_length=32, discount=1.,
                     animated=False, speedup=1):
    n_agents = len(env.agents)
    observations = [[] for _ in range(n_agents)]
    actions = [[] for _ in xrange(n_agents)]
    rewards = [[] for _ in xrange(n_agents)]
    agent_infos = [[] for _ in xrange(n_agents)]
    env_infos = [[] for _ in xrange(n_agents)]
    olist = env.reset()
    agent.reset()
    path_length = 0
    while path_length < max_path_length:
        alist = []
        for i, o in enumerate(olist):
            a, agent_info = agent.get_action(o)
            alist.append(a)
            observations[i].append(env.observation_space.flatten(o))
            actions[i].append(env.action_space.flatten(a))
            agent_infos[i].append(agent_info)

        next_olist, rlist, d, env_info = env.step(np.asarray(alist))
        for i, r in enumerate(rlist):
            rewards[i].append(r)
            env_infos[i].append(env_info)

        path_length += 1
        if d:
            break
        olist = next_olist

    chunked_observations = [[agent_obs[i:i + chunked_path_length]
                             for i in range(0, len(agent_obs), chunked_path_length)]
                            for agent_obs in observations]
    chunked_actions = [[agent_acts[i:i + chunked_path_length]
                        for i in range(0, len(agent_acts), chunked_path_length)]
                       for agent_acts in actions]
    chunked_rewards = [[agent_rew[i:i + chunked_path_length]
                        for i in range(0, len(agent_rew), chunked_path_length)]
                       for agent_rew in rewards]
    chunked_agent_infos = [[agent_info[i:i + chunked_path_length]
                            for i in range(0, len(agent_info), chunked_path_length)]
                           for agent_info in agent_infos]
    chunked_env_infos = [[env_info[i:i + chunked_path_length]
                          for i in range(0, len(env_info), chunked_path_length)]
                         for env_info in env_infos]

    trajs = [dict(observations=tensor_utils.stack_tensor_list(chunked_observations[i]),
                  actions=tensor_utils.stack_tensor_list(chunked_actions[i]),
                  rewards=tensor_utils.stack_tensor_list(chunked_rewards[i]),
                  agent_infos=tensor_utils.stack_tensor_dict_list(chunked_agent_infos[i]),
                  env_infos=tensor_utils.stack_tensor_dict_list(chunked_env_infos[i]),)
             for i in range(len(chunked_actions))]
    return trajs
