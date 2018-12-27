import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import os.path
import numpy as np
import matplotlib.pyplot as plt
import sys
import createData2

TIME_STEPS = 30
HIDDEN_UNITS = 1
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCH = 150


# def generate_lstm(X_train, y_train, X_test, y_test, save_flag=False, save_file_name='test.csv'):
def generate_lstm(X_train, y_train, save_flag=False, save_file_name='test.csv'):
    TRAIN_EXAMPLES = X_train.shape[0]
    # TEST_EXAMPLES = X_test.shape[0]
    graph = tf.Graph()
    with graph.as_default():
        # place hoder
        X_p = tf.placeholder(dtype=tf.float32, shape=(None, TIME_STEPS, 1), name="input_placeholder")
        y_p = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="pred_placeholder")

        lstm_cell = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS)
        print(lstm_cell.get_config())

        # initialize to zero
        init_state = lstm_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)

        # dynamic rnn
        outputs, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=X_p, initial_state=init_state, dtype=tf.float32)
        # print(outputs.shape)
        h = outputs[:, -1, :]
        s = states[:]
        # print(h.shape)

        # ---------------------------------define loss and optimizer----------------------------------#
        mse = tf.losses.mean_squared_error(labels=y_p, predictions=h)
        # print(loss.shape)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=mse)

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        sess.run(init)
        for epoch in range(1, EPOCH + 1):
            train_losses = []
            print("epoch:", epoch)
            for j in range(TRAIN_EXAMPLES // BATCH_SIZE):
                result, train_loss = sess.run(
                    fetches=(optimizer, mse),
                    feed_dict={
                        X_p: X_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE].reshape(-1, 30, 1),
                        y_p: y_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                    }
                )
                train_losses.append(train_loss)
            print("average training loss:", sum(train_losses) / len(train_losses))
        # if save_flag:
        #     save_model(save_file_name, sess)

        # test_losses = []
        # results = np.zeros(shape=(TEST_EXAMPLES, 1))
        # for j in range(TEST_EXAMPLES // BATCH_SIZE):
        #     result, state, test_loss = sess.run(
        #         fetches=(h, s, mse),
        #         feed_dict={
        #             X_p: X_test[j * BATCH_SIZE:(j + 1) * BATCH_SIZE].reshape(-1, 30, 1),
        #             y_p: y_test[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
        #         }
        #     )
        #     print(result)
        #     print('state: ')
        #     print(state)
        #     results[j * BATCH_SIZE:(j + 1) * BATCH_SIZE] = result
        #     test_losses.append(test_loss)
        # print("average test loss:", sum(test_losses) / len(test_losses))
        # plt.plot(range(1000), results[:1000, 0])
        # plt.show()


def save_model(model_name, sess):
    print('save model')
    model_path = '../../model/'
    model_path = os.path.join(model_path, model_name)
    m_saver = tf.train.Saver()
    m_saver.save(sess=sess, save_path=model_path)


if __name__ == '__main__':
    print(sys.path)
    train_data, train_label = createData2.create_data(start=0, stop=15, num=500, w=1)

    generate_lstm(train_data, train_label, True, 'test1.mode')
