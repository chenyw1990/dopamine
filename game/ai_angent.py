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
                 eval_mode=False,
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
            eval_mode=eval_mode,
            use_staging=use_staging)

    def _network_template(self, state):
        net = tf.cast(state, tf.float32)

        net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

        net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

        net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

        net = tf.reshape(net, [-1, 14 * 14 * 64])

        net = tf.layers.dense(inputs=net, units=512, activation=tf.nn.relu)

        q_values = tf.layers.dense(inputs=net, units=self.num_actions, activation=None)

        return self._get_network_type()(q_values)
