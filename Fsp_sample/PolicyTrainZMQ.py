from algos.fsp_sample import RARL
from regressors.gaussian_mlp_rarl_regressor import GaussianMLPRegressor

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from os import sys, path
sys.path.append(path.abspath(path.join(path.dirname(__file__), '..')))
from Ad05RTheta2RSteer2FR.julia2pythonZMQ_RARL import JustEgoEnv

from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.categorical_lstm_policy import CategoricalLSTMPolicy
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
import lasagne.nonlinearities as NL
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc import logger
import os.path as osp
import tensorflow as tf
import joblib
import numpy as np

from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces.box import Box
from rllab.envs.env_spec import EnvSpec
#stub(globals())

log_dir = "Data/col50tanhabs0"

tabular_log_file = osp.join(log_dir, "progress.csv")
text_log_file = osp.join(log_dir, "debug.log")
params_log_file = osp.join(log_dir, "params.json")
pkl_file = osp.join(log_dir, "params.pkl")

logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
prev_snapshot_dir = logger.get_snapshot_dir()
prev_mode = logger.get_snapshot_mode()
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode("gaplast")
logger.set_snapshot_gap(1000)
logger.set_log_tabular_only(False)
logger.push_prefix("[%s] " % "RARL-TF-1")

import samplers.lowlevel.rarl_parallel_sampler as parallel_sampler
parallel_sampler.initialize(n_parallel=1)
parallel_sampler.set_seed(0)

#env = normalize(MultilaneEnv(),1,True,True,0.001,0.001)
#env = normalize(MultilaneEnv())
env = TfEnv(JustEgoEnv(port=9410))

obs1_dim = 4
obs2_dim = 4
action1_dim = 2
action2_dim = 2

spec1 = EnvSpec(
                observation_space = Box(low=-np.ones(4), high=np.ones(4)),
                action_space = Box(low=-np.ones(2), high=np.ones(2)),
                )
spec2 = EnvSpec(
                observation_space = Box(low=-np.ones(4), high=np.ones(4)),
                action_space = Box(low=-np.ones(2), high=np.ones(2)),
                )

with tf.Session() as sess:
    policy1 = GaussianMLPPolicy(
        env_spec=spec1,
        name="RARLTFPolicy1",
        learn_std=True,
        init_std=0.1,
        output_nonlinearity=None,
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(256, 128, 64, 32),
        adaptive_std=True,
        std_hidden_sizes=(256, 128, 64, 32),
        std_hidden_nonlinearity=tf.nn.relu,
    )
    policy2 = GaussianMLPPolicy(
        env_spec=spec2,
        name="RARLTFPolicy2",
        learn_std=True,
        init_std=0.1,
        output_nonlinearity=None,
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(256, 128, 64, 32),
        adaptive_std=True,
        std_hidden_sizes=(256, 128, 64, 32),
        std_hidden_nonlinearity=tf.nn.relu,
    )
    sess.run(tf.global_variables_initializer())

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    baseline2 = LinearFeatureBaseline(env_spec=env.spec)

    algo = RARL(
        env=env,
        policy=policy1,
        policy2=policy2,
        obs1_dim = obs1_dim,
        obs2_dim = obs2_dim,
        action1_dim = action1_dim,
        action2_dim = action2_dim,
        baseline=baseline,
        baseline2=baseline2,
        batch_size=4096,
        max_path_length=210,
        n_itr=5001,
        N1 = 5,
        N2 = 2,
        discount=0.95,
        step_size=0.01,
        transfer=True,
        record_rewards=False,
        # optimizer_args=dict(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
        sample_policy_1=False,
        policy_path=log_dir+"/params",
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )        
    algo.train(sess)
