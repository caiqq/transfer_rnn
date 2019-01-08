"""Sequence-to-sequence model for human motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

import random

import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import rnn_cell_extensions
import Transfer_rnn_gate


class TransferSimpleModel(object):
    """Sequence-to-sequence model for human motion prediction"""

    def __init__(self,
                 session,
                 source_size,
                 model_source,
                 source_seq_len,
                 target_seq_len,
                 rnn_size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 dtype=tf.float32):
        """Create the model.

        Args:
          source_seq_len: length of the input sequence.
          target_seq_len: length of the target sequence.
          rnn_size: number of units in the rnn.
          num_layers: number of rnns to stack.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          summaries_dir: where to log progress for tensorboard.
          loss_to_use: [supervised, sampling_based]. Whether to use ground truth in
            each timestep to compute the loss after decoding, or to feed back the
            prediction from the previous time-step.
          residual_velocities: whether to use a residual connection that models velocities.
          dtype: the data type to use to store internal variables.
        """

        # Summary writers for train and test runs
        self.input_size = source_seq_len

        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.rnn_size = rnn_size

        self.size = rnn_size

        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        self.train_writer = tf.summary.FileWriter(os.path.normpath(os.path.join('../../summaries_dir/', 'train')))
        self.test_writer = tf.summary.FileWriter(os.path.normpath(os.path.join('../../summaries_dir/', 'test')))

        # === Create the RNN that will keep the state ===

        # cell = tf.contrib.rnn.GRUCell(self.rnn_size)
        # cell = tf.contrib.rnn.LSTMCell(self.rnn_size, state_is_tuple=True)

        # create multi layer rnn model
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.rnn_size[i]) for i in range(num_layers)])
            # cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(self.rnn_size, state_is_tuple=True)
            #                                     for _ in range(num_layers)])

        # === Transform the inputs ===
        with tf.name_scope("inputs_"):

            x_p = tf.placeholder(dtype=tf.float32, shape=(source_seq_len, 1), name="input_placeholder")
            y_p = tf.placeholder(dtype=tf.float32, shape=(1), name="pred_placeholder")

            self.encoder_inputs = x_p
            self.decoder_outputs = y_p

            # x_p = tf.split(x_p, source_seq_len, axis=0)
            # generate weights for sources-------------------------------------------
            self.w_h = []
            # self.v_h = []
            self.w_s = []
            # self.v_s = []
            self.w_b = []

            for source_index in range(source_size):
                w_h = vs.get_variable("TL_w_h_source_{0}".format(source_index), shape=[1, self.source_seq_len],
                                      initializer=tf.constant_initializer(0.1))
                # v_h = vs.get_variable("TL_v_h_source_{0}".format(FLAGS.transfer_actions), shape=[self.input_size, 1])
                w_s = vs.get_variable("TL_w_s_source_{0}".format(source_index), shape=[1, self.source_seq_len],
                                      initializer=tf.constant_initializer(0.1))
                # v_s = vs.get_variable("TL_v_s_source_{0}".format(FLAGS.transfer_actions), shape=[self.rnn_size, 1])
                w_b = vs.get_variable("TL_w_b_source_{0}".format(source_index), shape=[1],
                                      initializer=tf.constant_initializer(0.03))

                self.w_h.append(w_h)
                # self.v_h.append(v_h)
                self.w_s.append(w_s)
                # self.v_s.append(v_s)
                self.w_b.append(w_b)

            # for target:
            self.w_t_h = vs.get_variable("TL_w_h_target", shape=[1, self.source_seq_len],
                                         initializer=tf.constant_initializer(0.7))
            # self.v_t_h = vs.get_variable("TL_v_h_target", shape=[self.input_size, 1])
            self.w_t_s = vs.get_variable("TL_w_s_target", shape=[1, self.source_seq_len],
                                         initializer=tf.constant_initializer(0.7))
            # self.v_t_s = vs.get_variable("TL_v_s_target", shape=[self.rnn_size, 1])
            self.w_t_b = vs.get_variable("TL_w_b_target", shape=[1], initializer=tf.constant_initializer(0.05))
            session.run(tf.global_variables_initializer())

        # linear warapper after GRU, make the output of GRU has the same dimension as input for residual connedction
        # cell = rnn_cell_extensions.LinearSpaceDecoderWrapper(cell, self.input_size)

        transfer_model_dir = '../../model/train2/'
        model_path = os.path.normpath(os.path.join(transfer_model_dir, "model-" + str(1)))
        model_source.saver.restore(session, model_path)
        cell = model_source.cell

        # Store the outputs here
        outputs = []
        self.cell = cell

        # Define the loss function
        lf = None
        # if loss_to_use == "sampling_based":
        #   def lf(prev, i): # function for sampling_based loss
        #     return prev
        # elif loss_to_use == "supervised":
        #   pass
        # else:
        #   raise(ValueError, "unknown loss: %s" % loss_to_use)

        # Build the RNN
        with vs.variable_scope("basic_rnn_seq2seq"):
            self.outputs, self.states = Transfer_rnn_gate.static_rnn(session, source_size, model_source, self.w_h,
                                                                     self.w_s, self.w_b,
                                                                     self.w_t_h, self.w_t_s, self.w_t_b, self.cell, x_p,
                                                                     dtype=tf.float32)

            # self.outputs, self.states = Transfer_rnn_gate.static_lstm_rnn(session, source_size, model_source, self.w_h, self.w_s, self.w_b,
            #                                                          self.w_t_h, self.w_t_s, self.w_t_b, self.cell, x_p,
            #                                                          dtype=tf.float32)

            # outputs, self.states = Transfer_rnn_gate.static_rnn(session, model_source, self.w_h, self.w_s, self.w_b,
            #                                                     self.w_t_h, self.w_t_s, self.w_t_b, cell, x_p,
            #                                                     dtype=tf.float32)
        # self.outputs = outputs

        with tf.name_scope("loss_angles"):
            loss_angles = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(y_p, self.outputs[0][-1]))))

        self.loss = loss_angles
        self.loss_summary = tf.summary.scalar('loss/loss', self.loss)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()

        opt = tf.train.GradientDescentOptimizer(self.learning_rate)

        # opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.learning_rate_decay_factor,
        #                              beta2=self.learning_rate_decay_factor + 0.09, epsilon=1e-08, use_locking=False,
        #                              name='Adam')

        # Update all the trainable parameters
        gradients = tf.gradients(self.loss, params)

        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.gradient_norms = norm
        self.updates = opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

        # Keep track of the learning rate
        self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    def step(self, session, encoder_inputs, decoder_outputs,
             forward_only, srnn_seeds=False):
        """Run a step of the model feeding the given inputs.

        Args
          session: tensorflow session to use.
          encoder_inputs: list of numpy vectors to feed as encoder inputs.
          decoder_inputs: list of numpy vectors to feed as decoder inputs.
          decoder_outputs: list of numpy vectors that are the expected decoder outputs.
          forward_only: whether to do the backward step or only forward.
          srnn_seeds: True if you want to evaluate using the sequences of SRNN
        Returns
          A triple consisting of gradient norm (or None if we did not do backward),
          mean squared error, and the outputs.
        Raises
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        input_feed = {self.encoder_inputs: encoder_inputs,
                      self.decoder_outputs: decoder_outputs}

        # Output feed: depends on whether we do a backward step or not.
        if not srnn_seeds:
            if not forward_only:

                # Training step
                output_feed = [self.updates,  # Update Op that does SGD.
                               self.gradient_norms,  # Gradient norm.
                               self.loss,
                               self.loss_summary,
                               self.learning_rate_summary]

                outputs = session.run(output_feed, input_feed)
                return outputs[1], outputs[2], outputs[3], outputs[4]  # Gradient norm, loss, summaries

            else:
                # Validation step, not on SRNN's seeds
                output_feed = [self.loss,  # Loss for this batch.
                               self.loss_summary,
                               self.outputs]
                outputs = session.run(output_feed, input_feed)
                return outputs[0], outputs[1], outputs[2]  # No gradient norm
        else:
            # Validation on SRNN's seeds
            output_feed = [self.loss,  # Loss for this batch.
                           self.outputs,
                           self.loss_summary]

            outputs = session.run(output_feed, input_feed)

            return outputs[0], outputs[1], outputs[2]  # No gradient norm, loss, outputs.

    def get_batch(self, data):
        """Get a random batch of data from the specified bucket, prepare for step.

        Args
          data: a list of sequences of size n-by-d to fit the model to.
          actions: a list of the actions we are using
        Returns
          The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
          the constructed batches have the proper format to call step(...) later.
        """

        # Select entries at random
        all_keys = list(data.keys())
        chosen_keys = np.random.choice(len(all_keys), self.batch_size)

        # How many frames in total do we need?
        total_frames = self.source_seq_len + self.target_seq_len

        encoder_inputs = np.zeros((self.batch_size, self.source_seq_len - 1, self.input_size), dtype=float)
        decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)

        for i in xrange(self.batch_size):
            the_key = all_keys[chosen_keys[i]]

            # Get the number of frames
            n, _ = data[the_key].shape

            # Sample somewherein the middle
            idx = np.random.randint(16, n - total_frames)

            # Select the data around the sampled points
            data_sel = data[the_key][idx:idx + total_frames, :]

            # Add the data
            encoder_inputs[i, :, 0:self.input_size] = data_sel[0:self.source_seq_len - 1, :]
            decoder_outputs[i, :, 0:self.input_size] = data_sel[self.source_seq_len:, 0:self.input_size]

        return encoder_inputs, decoder_outputs

    # def find_indices_srnn(self, data, action):
    #     """
    #     Find the same action indices as in SRNN.
    #     See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    #     """
    #
    #     # Used a fixed dummy seed, following
    #     # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    #     SEED = 1234567890
    #     rng = np.random.RandomState(SEED)
    #
    #     subject = 5
    #     subaction1 = 1
    #     subaction2 = 2
    #
    #     T1 = data[(subject, action, subaction1, 'even')].shape[0]
    #     T2 = data[(subject, action, subaction2, 'even')].shape[0]
    #     prefix, suffix = 50, 100
    #
    #     idx = []
    #     for i in range(int(self.batch_size / 2)):
    #         idx.append(rng.randint(16, T1 - prefix - suffix))
    #         idx.append(rng.randint(16, T2 - prefix - suffix))
    #
    #     return idx

    # def get_batch_srnn(self, data):
    #     """
    #     Get a random batch of data from the specified bucket, prepare for step.
    #
    #     Args
    #       data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
    #         v=nxd matrix with a sequence of poses
    #       action: the action to load data from
    #     Returns
    #       The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
    #       the constructed batches have the proper format to call step(...) later.
    #     """
    #     frames = {}
    #     frames[action] = self.find_indices_srnn(data, action)
    #
    #     batch_size = self.batch_size  # we always evaluate 8 seeds
    #     subject = 5  # we always evaluate on subject 5
    #     source_seq_len = self.source_seq_len
    #     target_seq_len = self.target_seq_len
    #
    #     seeds = [(action, (i % 2) + 1, frames[action][i]) for i in range(batch_size)]
    #
    #     encoder_inputs = np.zeros((batch_size, source_seq_len - 1, self.input_size), dtype=float)
    #     decoder_outputs = np.zeros((batch_size, target_seq_len, self.input_size), dtype=float)
    #
    #     # Compute the number of frames needed
    #     total_frames = source_seq_len + target_seq_len
    #
    #     # Reproducing SRNN's sequence subsequence selection as done in
    #     # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
    #     for i in xrange(batch_size):
    #         _, subsequence, idx = seeds[i]
    #         idx = idx + 50
    #
    #         data_sel = data[(subject, action, subsequence, 'even')]
    #
    #         data_sel = data_sel[(idx - source_seq_len):(idx + target_seq_len), :]
    #
    #         encoder_inputs[i, :, :] = data_sel[0:source_seq_len - 1, :]
    #         decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]
    #
    #     return encoder_inputs, decoder_outputs
