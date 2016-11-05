import numpy as np
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger
from rllab.sampler.base import BaseSampler
from rllab.sampler import parallel_sampler, ma_sampler
from rllab.sampler.stateful_pool import singleton_pool
import tensorflow as tf
import itertools


def worker_init_tf(G):
    G.sess = tf.Session()
    G.sess.__enter__()


def worker_init_tf_vars(G):
    G.sess.run(tf.initialize_all_variables())


class BatchMASampler(BaseSampler):

    def start_worker(self):
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf)
        if hasattr(self.algo, 'policies'):
            ma_sampler.populate_task(self.algo.env, self.algo.policies, self.algo.ma_mode)
        else:
            ma_sampler.populate_task(self.algo.env, self.algo.policy, self.algo.ma_mode)
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf_vars)

    def shutdown_worker(self):
        ma_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr):
        if self.algo.ma_mode == 'concurrent':
            cur_policy_params = [policy.get_param_values() for policy in self.algo.policies]
        else:
            cur_policy_params = self.algo.policy.get_param_values()
        if hasattr(self.algo.env, "get_param_values"):
            cur_env_params = self.algo.env.get_param_values()
        else:
            cur_env_params = None
        paths = ma_sampler.sample_paths(
            policy_params=cur_policy_params,
            env_params=cur_env_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            ma_mode=self.algo.ma_mode,
            scope=self.algo.scope,)
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated

    def process_samples(self, itr, paths):
        if self.algo.ma_mode == 'centralized':
            return super().process_samples(itr, paths)
        elif self.algo.ma_mode == 'decentralized':
            return super().process_samples(itr, list(itertools.chain.from_iterable(paths)))
        elif self.algo.ma_mode == 'concurrent':
            processed_samples = []
            for ps, policy, baseline in zip(paths, self.algo.policies, self.algo.baselines):
                baselines = []
                returns = []

                if hasattr(baseline, "predict_n"):
                    all_path_baselines = baseline.predict_n(ps)
                else:
                    all_path_baselines = [baseline.predict(path) for path in ps]

                for idx, path in enumerate(ps):
                    path_baselines = np.append(all_path_baselines[idx], 0)
                    deltas = path["rewards"] + \
                             self.algo.discount * path_baselines[1:] - \
                             path_baselines[:-1]
                    path["advantages"] = special.discount_cumsum(deltas, self.algo.discount *
                                                                 self.algo.gae_lambda)
                    path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
                    baselines.append(path_baselines[:-1])
                    returns.append(path["returns"])

                ev = special.explained_variance_1d(
                    np.concatenate(baselines), np.concatenate(returns))

                if not policy.recurrent:
                    observations = tensor_utils.concat_tensor_list(
                        [path["observations"] for path in ps])
                    actions = tensor_utils.concat_tensor_list([path["actions"] for path in ps])
                    rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in ps])
                    returns = tensor_utils.concat_tensor_list([path["returns"] for path in ps])
                    advantages = tensor_utils.concat_tensor_list(
                        [path["advantages"] for path in ps])
                    env_infos = tensor_utils.concat_tensor_dict_list(
                        [path["env_infos"] for path in ps])
                    agent_infos = tensor_utils.concat_tensor_dict_list(
                        [path["agent_infos"] for path in ps])

                    if self.algo.center_adv:
                        advantages = util.center_advantages(advantages)

                    if self.algo.positive_adv:
                        advantages = util.shift_advantages_to_positive(advantages)

                    average_discounted_return = \
                                                np.mean([path["returns"][0] for path in ps])

                    undiscounted_returns = [sum(path["rewards"]) for path in ps]

                    ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))

                    samples_data = dict(
                        observations=observations,
                        actions=actions,
                        rewards=rewards,
                        returns=returns,
                        advantages=advantages,
                        env_infos=env_infos,
                        agent_infos=agent_infos,
                        ps=ps,)
                else:
                    max_path_length = max([len(path["advantages"]) for path in ps])

                    # make all ps the same length (pad extra advantages with 0)
                    obs = [path["observations"] for path in ps]
                    obs = tensor_utils.pad_tensor_n(obs, max_path_length)

                    if self.algo.center_adv:
                        raw_adv = np.concatenate([path["advantages"] for path in ps])
                        adv_mean = np.mean(raw_adv)
                        adv_std = np.std(raw_adv) + 1e-8
                        adv = [(path["advantages"] - adv_mean) / adv_std for path in ps]
                    else:
                        adv = [path["advantages"] for path in ps]

                    adv = np.asarray([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

                    actions = [path["actions"] for path in ps]
                    actions = tensor_utils.pad_tensor_n(actions, max_path_length)

                    rewards = [path["rewards"] for path in ps]
                    rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

                    returns = [path["returns"] for path in ps]
                    returns = tensor_utils.pad_tensor_n(returns, max_path_length)

                    agent_infos = [path["agent_infos"] for path in ps]
                    agent_infos = tensor_utils.stack_tensor_dict_list(
                        [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos])

                    env_infos = [path["env_infos"] for path in ps]
                    env_infos = tensor_utils.stack_tensor_dict_list(
                        [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos])

                    valids = [np.ones_like(path["returns"]) for path in ps]
                    valids = tensor_utils.pad_tensor_n(valids, max_path_length)

                    average_discounted_return = \
                                                np.mean([path["returns"][0] for path in ps])

                    undiscounted_returns = [sum(path["rewards"]) for path in ps]

                    ent = np.sum(policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

                    samples_data = dict(
                        observations=obs,
                        actions=actions,
                        advantages=adv,
                        rewards=rewards,
                        returns=returns,
                        valids=valids,
                        agent_infos=agent_infos,
                        env_infos=env_infos,
                        ps=ps,)

                logger.log("fitting baseline...")
                if hasattr(baseline, 'fit_with_samples'):
                    baseline.fit_with_samples(ps, samples_data)
                else:
                    baseline.fit(ps)
                    logger.log("fitted")

                logger.record_tabular('Iteration', itr)
                logger.record_tabular('AverageDiscountedReturn', average_discounted_return)
                logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
                logger.record_tabular('ExplainedVariance', ev)
                logger.record_tabular('NumTrajs', len(ps))
                logger.record_tabular('Entropy', ent)
                logger.record_tabular('Perplexity', np.exp(ent))
                logger.record_tabular('StdReturn', np.std(undiscounted_returns))
                logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
                logger.record_tabular('MinReturn', np.min(undiscounted_returns))

                processed_samples.append(samples_data)

            return processed_samples
