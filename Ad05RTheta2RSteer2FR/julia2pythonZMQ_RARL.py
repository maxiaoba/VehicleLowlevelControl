import zmq
import gym
from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np
from rllab.core.serializable import Serializable
from scipy.stats import norm

# def get_pdf(x):
#     if x>1 or x<-1:
#         return 0
#     else:
#         return 0.5

class ZMQConnection:

    def __init__(self, ip, port):
        self._ip = ip
        self._port = port

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect("tcp://{}:{}".format(ip, port))

    @property
    def socket(self):
        return self._socket

    def sendreq(self, msg):
        self.socket.send_json(msg)
        respmsg = self.socket.recv_json()
        return respmsg


class JustEgoEnv(Env, Serializable):

    def __init__(self, ip='127.0.0.1', port=9397):
        self._conn = ZMQConnection(ip, port)
        self.prev_action = 0.

        Serializable.quick_init(self, locals())

    def step(self, Action):
        action = Action['action']
        action = np.tanh(action)
        data = self._conn.sendreq({"cmd": "step", "action": action.tolist()})
        prev_action = self.prev_action
        self.prev_action = action
        assert 'obs' in data
        assert 'rew' in data
        assert 'done' in data
        assert 'info' in data
        reward = data['rew']
        if Action['policy_num'] == 2:
            if reward == -50:
                reward = -reward
            else:
                dacc = action[2]
                dsteer = action[3]
                reward = -0.01*(np.abs(dacc)+np.abs(dsteer))

        return Step(observation=data['obs'], reward=reward, done=data['done'])

    def reset(self):
        data = self._conn.sendreq({"cmd": "reset"})
        self.prev_action = 0.
        assert 'obs' in data
        return np.array(data['obs'])

    def render(self):
        data = self._conn.sendreq({"cmd": "render"})
        return

    @property
    def observation_space(self):
        return Box(low=-np.ones(4), high=np.ones(4))

    @property
    def action_space(self):
        return Box(low=-np.ones(2), high=np.ones(2))

    def close(self):
        self._conn.socket.close()


if __name__ == '__main__':
    env = JustEgoEnv()
    obs = env.reset()
    while True:
        action = np.array(env.action_space.sample())
        print(action)
        ob, reward, done, _ = env.step(action)

        print("s ->{}".format(obs))
        print("a ->{}".format(action))
        print("sp->{}".format(ob))
        print("r ->{}".format(reward))

        obs = ob
        if done:
            break

    env.close()
