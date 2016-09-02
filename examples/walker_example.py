import os.path as osp
import gym

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.envs.gym_env import GymEnv
import rllab.misc.logger as logger

env = normalize(GymEnv('BipedalWalker-v2'))

#policy = GaussianMLPPolicy(
#    env_spec=env.spec,
#    # The neural network policy should have two hidden layers, each with 32 hidden units.
#    hidden_sizes=(64, 64)
#)

policy = GaussianGRUPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32,)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

# logger
LOG_DIR = 'walker_gru_test'

tabular_log_file = osp.join(LOG_DIR, 'progress.csv')
text_log_file = osp.join(LOG_DIR, 'debug.log')
params_log_file = osp.join(LOG_DIR, 'params.json')

logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
logger.set_snapshot_dir(LOG_DIR)
logger.set_snapshot_mode('last')
logger.set_log_tabular_only(False)
logger.push_prefix("[%s] " % 'Walker')

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=1200,
    max_path_length=500,
    n_itr=500,
    discount=0.99,
    step_size=0.01,
    mode='centralized'
)
algo.train()
