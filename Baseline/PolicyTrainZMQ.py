import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

from algos.trpo_transfer import TRPO_t
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from os import sys, path
sys.path.append(path.abspath(path.join(path.dirname(__file__), '..')))
from CommonFiles.julia2pythonZMQ_Baseline import JustEgoEnv
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
import lasagne.nonlinearities as NL
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc import logger
import os.path as osp
import tensorflow as tf
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
#stub(globals())

log_dir = "Data/Ad05RTheta2RSteer2FR"

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
logger.push_prefix("[%s] " % "JustEgo_Baseline")

from rllab.sampler import parallel_sampler
parallel_sampler.initialize(n_parallel=1)
parallel_sampler.set_seed(0)

env = TfEnv(JustEgoEnv(port=9411))

obs1_dim = 4
action1_dim = 2

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    name="BaselinePolicy",
    learn_std=True,
    init_std=0.1,
    output_nonlinearity=None,
    hidden_nonlinearity=tf.nn.relu,
    hidden_sizes=(256, 128, 64, 32),
    adaptive_std=True,
    std_hidden_sizes=(256, 128, 64, 32),
    std_hidden_nonlinearity=tf.nn.relu,
)


baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO_t(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4096,
    max_path_length=210,
    n_itr=15000,
    discount=0.95,
    step_size=0.01,
    sampler_cls=BatchSampler,
    record_rewards=True,
    transfer=False,
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)

algo.train()

