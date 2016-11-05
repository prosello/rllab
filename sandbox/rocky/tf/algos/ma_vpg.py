from rllab.misc import logger
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.batch_ma_polopt import BatchMAPolopt
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.misc import tensor_utils
from rllab.core.serializable import Serializable
import tensorflow as tf
import copy


class MAVPG(BatchMAPolopt, Serializable):
    """
    Multi-agent Vanilla Policy Gradient.
    """

    def __init__(self, env, policy_or_policies, baseline_or_baselines, optimizer=None,
                 optimizer_args=None, **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1,)
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
            optimizer = FirstOrderOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.opt_info = None
        super(MAVPG, self).__init__(env=env, policy_or_policies=policy_or_policies,
                                    baseline_or_baselines=baseline_or_baselines, **kwargs)

    def opt_helper(self, policy, optimizer):
        is_recurrent = int(policy.recurrent)

        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,)
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,)
        advantage_var = tensor_utils.new_tensor(
            name='advantage',
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
        logli = dist.log_likelihood_sym(action_var, dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        if is_recurrent:
            surr_obj = -tf.reduce_sum(logli * advantage_var * valid_var) / tf.reduce_sum(valid_var)
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            max_kl = tf.reduce_max(kl * valid_var)
        else:
            surr_obj = -tf.reduce_mean(logli * advantage_var)
            mean_kl = tf.reduce_mean(kl)
            max_kl = tf.reduce_max(kl)

        input_list = [obs_var, action_var, advantage_var] + state_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        optimizer.update_opt(loss=surr_obj, target=policy, inputs=input_list)

        f_kl = tensor_utils.compile_function(
            inputs=input_list + old_dist_info_vars_list,
            outputs=[mean_kl, max_kl],)
        opt_info = dict(f_kl=f_kl,)
        return opt_info

    @overrides
    def init_opt(self):
        if self.ma_mode == 'concurrent':
            self.opt_infos = []
            self.optimizers = []
            for idx, policy in enumerate(self.policies):
                with tf.variable_scope('agent_%d' % idx):
                    optimizer = copy.deepcopy(self.optimizer)
                    opt_info = self.opt_helper(policy, optimizer)
                    self.opt_infos.append(opt_info)
                    self.optimizers.append(optimizer)
        else:
            self.opt_info = self.opt_helper(self.policy, self.optimizer)

    def optimize_policy_helper(self, samples_data, policy, optimizer, opt_info):
        logger.log("optimizing policy")
        inputs = ext.extract(samples_data, "observations", "actions", "advantages")
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in policy.state_info_keys]
        inputs += tuple(state_info_list)
        if policy.recurrent:
            inputs += (samples_data["valids"],)
        dist_info_list = [agent_infos[k] for k in policy.distribution.dist_info_keys]
        loss_before = optimizer.loss(inputs)
        optimizer.optimize(inputs)
        loss_after = optimizer.loss(inputs)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)

        mean_kl, max_kl = opt_info['f_kl'](*(list(inputs) + dist_info_list))
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('MaxKL', max_kl)

    @overrides
    def optimize_policy(self, itr, samples_data):
        if self.ma_mode == 'concurrent':
            for idx, policy in enumerate(self.policies):
                with logger.prefix('agent #%d | ' % idx):
                    self.optimize_policy_helper(samples_data[idx], policy, self.optimizers[idx],
                                                self.opt_infos[idx])
        else:
            self.optimize_policy_helper(samples_data, self.policy, self.optimizer, self.opt_info)

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
