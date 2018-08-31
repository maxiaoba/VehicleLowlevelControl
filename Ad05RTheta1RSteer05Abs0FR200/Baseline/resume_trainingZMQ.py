from __future__ import print_function
from __future__ import absolute_import

from rllab.sampler.utils import rollout
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
import argparse
import joblib
import uuid
import random
import numpy as np
import json
from THEANO import logger
from rllab.misc.instrument import to_local_command
import os
import signal
import subprocess
import tensorflow as tf
import os.path as osp

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.algos.trpo import TRPO
from TF.trpo_transfer import TRPO_t
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler

filename = str(uuid.uuid4())

if __name__ == "__main__":
    ###
    #signal.signal(signal.SIGCHLD, segment_handler)
    log_dir = "/Users/xiaobaima/Dropbox/SISL/AST/rllab/data/local/experiment/BaselineZMQ"
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
    with tf.Session() as sess:
        data = joblib.load(pkl_file)
        env = data['env']
        policy = data['policy']
        idx = data['itr']
        baseline = LinearFeatureBaseline(env_spec=env.spec)
        if 'rewards' in data.keys():
            print("already has rewards~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            rewards = data['rewards']
            algo = TRPO_t(env=env, 
                policy=policy, 
                baseline=baseline, 
                start_itr=idx,    
                batch_size=4000,
                max_path_length=210,
                n_itr=5000,
                discount=0.95,
                step_size=0.01,
                sampler_cls=BatchSampler,
                record_rewards=True,
                rewards=rewards,
                transfer=True,
            )
        else:
            print("no rewards~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            algo = TRPO_t(env=env, 
                policy=policy, 
                baseline=baseline, 
                start_itr=idx,    
                batch_size=4000,
                max_path_length=210,
                n_itr=5000,
                discount=0.95,
                step_size=0.01,
                sampler_cls=BatchSampler,
                record_rewards=True,
                transfer=True
            )
        algo.train(sess)
        #assert 'algo' in data
        #algo = data['algo']
        #algo.train()
