import tensorflow as tf
import os
import SimpleModel
import mTest
import createData2

max_gradient_norm = 5
batch_size = 128

seq_length_in = 30
seq_length_out = 1

learning_rate = 0.09
learning_rate_decay_factor = 0.99
learning_rate_step = 150

num_layers = 1
size = 1

load = 0


def load_model():
    with tf.Session() as sess:

        model_source = SimpleModel.SimpleModel(
            seq_length_in,
            seq_length_out,
            size,  # hidden layer size
            num_layers,
            max_gradient_norm,
            batch_size,
            learning_rate,
            learning_rate_decay_factor,
            dtype=tf.float32)

        transfer_model_dir = '../../model/train/'
        model_path = os.path.normpath(os.path.join(transfer_model_dir, "model-" + str(2)))

        model_source.saver.restore(sess, model_path)

        data_file_name = 'data_period150_w1_mu0_sigma002'

        datas = createData2.read_original_data(data_file_name)
        datas = createData2.normalization_data(datas)
        data_len = 450
        print('data len: ', data_len)
        # data_train = datas[0:data_len]
        data_test = datas[data_len:]
        # train_data, train_label = createData2.generate(data_train)
        test_data, test_label = createData2.generate(data_test)

        mTest.test(model_source, sess, test_data, test_label)


if __name__ == '__main__':
    load_model()
