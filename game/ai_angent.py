from tensorflow.contrib import slim

from dopamine.agents.dqn import dqn_agent
import tensorflow as tf
import numpy as np

OBSERVATION_SHAPE = 14
STACK_SIZE = 4


class AIAgent(dqn_agent.DQNAgent):
    def __init__(self,
                 sess,
                 num_actions,
                 min_replay_history=32,
                 update_period=4,
                 target_update_period=8000,
                 epsilon_fn=dqn_agent.linearly_decaying_epsilon,
                 epsilon_train=0.01,
                 epsilon_eval=0.001,
                 epsilon_decay_period=250000,
                 tf_device='/cpu:*',
                 use_staging=True):
        super(AIAgent, self).__init__(
            sess=sess,
            num_actions=num_actions,
            min_replay_history=min_replay_history,
            update_period=update_period,
            target_update_period=target_update_period,
            epsilon_fn=epsilon_fn,
            epsilon_train=epsilon_train,
            epsilon_eval=epsilon_eval,
            epsilon_decay_period=epsilon_decay_period,
            tf_device=tf_device,
            use_staging=use_staging)

    def _network_template(self, state):
        net = tf.cast(state, tf.float32)

        net = slim.conv2d(net, 32, [5, 5], stride=1)
        net = slim.conv2d(net, 64, [3, 3], stride=1)
        net = slim.conv2d(net, 64, [3, 3], stride=1)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 512)
        q_values = slim.fully_connected(net, self.num_actions, activation_fn=None)
        return self._get_network_type()(q_values)
