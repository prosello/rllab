from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.algos.batch_ma_polopt import BatchMAPolopt
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
import copy


class MANPO(BatchMAPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(self, optimizer=None, optimizer_args=None, step_size=0.01, **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        super(MANPO, self).__init__(**kwargs)

    def opt_helper(self, policy, optimizer):
        is_recurrent = int(policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,)
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,)
        advantage_var = tensor_utils.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=tf.float32,)
        dist = policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape),
                              name='old_%s' % k)
            for k, shape in dist.dist_info_specs
        }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in policy.state_info_specs
        }
        state_info_vars_list = [state_info_vars[k] for k in policy.state_info_keys]

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

        dist_info_vars = policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if is_recurrent:
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            surr_loss = -tf.reduce_sum(lr * advantage_var * valid_var) / tf.reduce_sum(valid_var)
        else:
            mean_kl = tf.reduce_mean(kl)
            surr_loss = -tf.reduce_mean(lr * advantage_var)

        input_list = [
            obs_var,
            action_var,
            advantage_var,
        ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        optimizer.update_opt(loss=surr_loss, target=policy,
                             leq_constraint=(mean_kl, self.step_size), inputs=input_list,
                             constraint_name="mean_kl")

    @overrides
    def init_opt(self):
        if self.ma_mode == 'concurrent':
            self.optimizers = []
            for idx, policy in enumerate(self.policies):
                with tf.variable_scope('agent_%d' % idx):
                    optimizer = copy.deepcopy(self.optimizer)
                    self.opt_helper(policy, optimizer)
                    self.optimizers.append(optimizer)
        else:
            self.opt_helper(self.policy, self.optimizer)
        return dict()

    def optimize_policy_helper(self, samples_data, policy, optimizer):
        all_input_values = tuple(ext.extract(samples_data, "observations", "actions", "advantages"))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if policy.recurrent:
            all_input_values += (samples_data["valids"],)
        logger.log("Computing loss before")
        loss_before = optimizer.loss(all_input_values)
        logger.log("Computing KL before")
        mean_kl_before = optimizer.constraint_val(all_input_values)
        logger.log("Optimizing")
        optimizer.optimize(all_input_values)
        logger.log("Computing KL after")
        mean_kl = optimizer.constraint_val(all_input_values)
        logger.log("Computing loss after")
        loss_after = optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)

    @overrides
    def optimize_policy(self, itr, samples_data):
        if self.ma_mode == 'concurrent':
            for idx, policy in enumerate(self.policies):
                with logger.prefix('agent #%d | ' % idx):
                    self.optimize_policy_helper(samples_data[idx], policy, self.optimizers[idx])
        else:
            self.optimize_policy_helper(samples_data, self.policy, self.optimizer)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        if self.ma_mode == 'concurrent':
            return dict(
                itr=itr,
                policies=self.policies,
                baselines=self.baselines,
                env=self.env,)
        else:
            return dict(
                itr=itr,
                policy=self.policy,
                baseline=self.baseline,
                env=self.env,)
