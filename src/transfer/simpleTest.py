import tensorflow as tf
import os
import SimpleModel
import mTest
import createData2

max_gradient_norm = 5
batch_size = 30

seq_length_in = 30
seq_length_out = 1

learning_rate = 0.09
learning_rate_decay_factor = 0.99
learning_rate_step = 150

num_layers = 1
size = 1

load = 0


def load_model():
    data_file_name = 'data_period150_w1_mu0_sigma002'

    datas = createData2.read_original_data(data_file_name)
    datas = createData2.normalization_data(datas)
    data_len = 450
    print('data len: ', data_len)
    # data_train = datas[0:data_len]
    data_test = datas[data_len:data_len+35]
    # train_data, train_label = createData2.generate(data_train)
    test_data, test_label = createData2.generate(data_test)

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
        for i in range(2):
            print('i: ', i)
            model_path = os.path.normpath(os.path.join(transfer_model_dir, "model-" + str(i+1)))

            model_source.saver.restore(sess, model_path)
            output_datas = []
            losses = []
            # Tstate = model_source.cell.zero_state(batch_size, dtype=float)
            # Tstate = Tstate[:, :, 0]
            for j in range(len(test_label) - 1):
                encoder_inputs = test_data[j:(j + 1)].reshape(30, 1),
                decoder_outputs = test_label[j:(j + 1)]

                # call_cell_source = lambda: model_source.cell(encoder_inputs[0], Tstate)
                # (Toutput, Tstate) = call_cell_source()

                step_loss, loss_summary, output_data = model_source.step(sess, encoder_inputs[0], decoder_outputs[0],
                                                                  forward_only=True)
                output_datas.append(output_data[0][-1])
                losses.append(step_loss)
                print('output_data: ', output_data[0][-1])
            # mean_loss = sum(losses) / len(losses)
            # print('mean_loss: ', mean_loss)
        # print(test_label)
        # print(len(test_label))
        # print(test_data)
        # mTest.test(model_source, sess, test_data, test_label)


if __name__ == '__main__':
    load_model()

