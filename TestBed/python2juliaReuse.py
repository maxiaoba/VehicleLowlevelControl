from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
import joblib
import tensorflow as tf
import numpy as np


class PolicyContainer():
    def __init__(self):
        self.sess = tf.Session()

    def reset_policy(self,policy_path=None,policy_baseline_path=None):
        print("reset_policy")
        tf.reset_default_graph()
        self.sess.close()
        self.sess = tf.Session()
        self.sess.__enter__()
        if policy_path is not None:
            print("policy_path is not None")
            with tf.variable_scope("rarl",reuse=None):
            # with tf.variable_scope(name):
                obj = joblib.load(policy_path)
                if 'policy' in obj.keys():
                    self.policy = obj['policy']
                if 'policy2' in obj.keys():
                    self.policy2 = obj['policy2']
                if 'rewards' in obj.keys():
                    self.rewards = obj['rewards']
                if 'regressor1' in obj.keys():
                    self.regressor1 = obj['regressor1']
                    self.regressor1.reload_initialize(self.sess)
                    self.regressor2 = obj['regressor2']
                    self.regressor2.reload_initialize(self.sess)

        if policy_baseline_path is not None:
            print("policy_baseline_path is not None")
            with tf.variable_scope("baseline",reuse=None):
                obj2 = joblib.load(policy_baseline_path)
                self.policy_baseline = obj2['policy']
                
    def resetPolicy(self,policy=0):
        if policy == 0:
            self.policy_baseline.reset()
        elif policy == 1:
            self.policy.reset()
        elif policy == 2:
            self.polic2.reset()

    def getAction(self,state,policy_num=0):
        # action, dist = sess.run(policy.get_action(np.array(state)))
        # assert isinstance(state,list)
        if policy_num == 0:
            action1, dist1 = self.policy.get_action(np.array(state))
            action2, dist2 = self.policy2.get_action(np.array(state))
            return action1,action2
        elif policy_num == 1:
            action1, dist1 = self.policy.get_action(np.array(state))
            return action1
        elif policy_num == 2:
            action2, dist2 = self.policy2.get_action(np.array(state))
            return action2

    def getAction_regressor(self,state,policy_num=0):
        if policy_num == 0:
            action1, dist1 = self.regressor1.get_action(np.array(state))
            action2, dist2 = self.regressor2.get_action(np.array(state))
            return action1,action2
        elif policy_num == 1:
            action1, dist1 = self.regressor1.get_action(np.array(state))
            return action1
        elif policy_num == 2:
            action2, dist2 = self.regressor2.get_action(np.array(state))
            return action2

    def getAction_baseline(self,state):
        # action, dist = sess.run(policy.get_action(np.array(state)))
        # assert isinstance(state,list)
        action1, dist1 = self.policy_baseline.get_action(np.array(state))
        return action1

    def getAction_just(self,state):
        action, dist = self.policy.get_action(np.array(state))
        return action

    def getPolicy(self):
        return self.policy

    def getActionDistribution(self,state):
        # action, dist = sess.run(policy.get_action(np.array(state)))
        # assert isinstance(state,list)
        action1, dist1 = self.policy.get_action(np.array(state))
        action2, dist2 = self.policy2.get_action(np.array(state))
        return dist1,dist2

    def getActionDistribution_just(self,state):
        action1, dist1 = self.policy.get_action(np.array(state))
        return dist1

    def getActionDistribution_baseline(self,state):
        # action, dist = sess.run(policy.get_action(np.array(state)))
        # assert isinstance(state,list)
        action1, dist1 = self.policy_baseline.get_action(np.array(state))
        return dist1

def getPolicyContainer():
    return PolicyContainer()