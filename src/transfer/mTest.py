import tensorflow as tf
import os
import time
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import createData2
import SimpleModel
import TransferSimpleModel


train_dir = os.path.normpath(os.path.join('../../model/', 'train4/'))


def create_model(session, seq_length_in, seq_length_out, size, num_layers, max_gradient_norm, batch_size,
                 learning_rate, learning_rate_decay_factor, load=0, transfer=0, source_size=1):
    if transfer > 0:
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

        print('model_source finished!')
        model_target = TransferSimpleModel.TransferSimpleModel(
            session,
            source_size,
            model_source,
            seq_length_in,
            seq_length_out,
            size,  # hidden layer size
            num_layers,
            max_gradient_norm,
            batch_size,
            learning_rate,
            learning_rate_decay_factor,
            dtype=tf.float32)
        print('model_target finished!')
        session.run(tf.global_variables_initializer())
        return model_target

    else:
        model = SimpleModel.SimpleModel(
            seq_length_in,
            seq_length_out,
            size,  # hidden layer size
            num_layers,
            max_gradient_norm,
            batch_size,
            learning_rate,
            learning_rate_decay_factor,
            dtype=tf.float32)

        if load <= 0:
            print("Creating model with fresh parameters.")
            session.run(tf.global_variables_initializer())
            return model

        ckpt = tf.train.get_checkpoint_state(train_dir, latest_filename="checkpoint")
        print("train_dir", train_dir)

        if ckpt and ckpt.model_checkpoint_path:
            # Check if the specific checkpoint exists
            if load > 0:
                if os.path.isfile(os.path.join(train_dir, str(load))):
                    ckpt_name = os.path.normpath(os.path.join(os.path.join(train_dir, str(load))))
                else:
                    raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(load))
            else:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            print("Loading model {0}".format(ckpt_name))
            model.saver.restore(session, ckpt.model_checkpoint_path)
            return model
        else:
            print("Could not find checkpoint. Aborting.")
            raise (ValueError, "Checkpoint {0} does not seem to exist".format(ckpt.model_checkpoint_path))

        return model

def create_source_datas(train_data, train_label, transfer, test_data, test_label, source_index=1):

    max_gradient_norm = 5
    batch_size = 128

    seq_length_in = 30
    seq_length_out = 1

    learning_rate = 0.07
    learning_rate_decay_factor = 0.99

    num_layers = 1
    size = 1

    load = 0

    source_size = source_index

    outputs = []
    with tf.Session() as sess:
        print("Creating %d layers." % (num_layers))

        model = create_model(sess, seq_length_in, seq_length_out, size, num_layers, max_gradient_norm,
                             batch_size, learning_rate, learning_rate_decay_factor, load, transfer, source_size)
        model.train_writer.add_graph(sess.graph)
        print("Model created")

        if transfer == 1:

            for source_index_num in range(source_size):
                transfer_model_dir = os.path.normpath(os.path.join('../../model/', 'train/'))
                model_path = os.path.normpath(os.path.join(transfer_model_dir, "model-" + str(source_index_num + 1)))

                model.model_source.saver.restore(sess, model_path)

                sub_outputs = []
                for j in range(len(train_data) - seq_length_in):
                    encoder_inputs = train_data[j:(j + 1)].reshape(30, 1),
                    decoder_outputs = train_label[j:(j + 1)]
                    step_loss, loss_summary, output_data = model.model_source.step(sess, encoder_inputs[0], decoder_outputs[0],
                                                                         forward_only=True)
                    sub_outputs.append(output_data[0].reshape(30))
                outputs.append(sub_outputs)

            file_name = 'train_datas'
            # file_name = 'test_datas'
            file_full_path = os.path.join('../../source_datas/', 'test_datas1')
            data_frame = pd.DataFrame(data=outputs[0])
            data_frame.to_csv(file_full_path, encoding='utf-8')

            file_full_path2 = os.path.join('../../source_datas/', 'test_datas2')
            data_frame = pd.DataFrame(data=outputs[1])
            data_frame.to_csv(file_full_path2, encoding='utf-8')

            file_full_path3 = os.path.join('../../source_datas/', 'test_datas3')
            data_frame = pd.DataFrame(data=outputs[2])
            data_frame.to_csv(file_full_path3, encoding='utf-8')

            source_datas = pd.read_csv(file_full_path)
            print(source_datas.columns)
            data = source_datas.iloc[0][1:].tolist()
            data = np.reshape(data, (30, 1))
            print(data)
    #     sys.stdout.flush()
    # return outputs



def train(train_data, train_label, transfer, test_data, test_label, source_index=1):
    epoch = 750
    # epoch = 250
    max_gradient_norm = 5
    batch_size = 128

    seq_length_in = 30
    seq_length_out = 1

    # learning_rate = 0.07
    learning_rate = 0.09
    learning_rate_decay_factor = 0.992
    learning_rate_step = 150

    num_layers = 1
    size = 1

    load = 0

    source_size = source_index

    file_full_path_1 = os.path.join('../../source_datas/', 'train_datas1')
    file_full_path_2 = os.path.join('../../source_datas/', 'train_datas2')
    file_full_path_3 = os.path.join('../../source_datas/', 'train_datas3')
    source_datas_1 = pd.read_csv(file_full_path_1)
    source_datas_2 = pd.read_csv(file_full_path_2)
    source_datas_3 = pd.read_csv(file_full_path_3)

    with tf.Session() as sess:
        print("Creating %d layers." % (num_layers))

        model = create_model(sess, seq_length_in, seq_length_out, size, num_layers, max_gradient_norm,
                             batch_size, learning_rate, learning_rate_decay_factor, load, transfer, source_size)
        model.train_writer.add_graph(sess.graph)
        print("Model created")

        # === Training step ===
        current_step = 0
        loss_list = []
        for epoch_index in range(epoch):
            current_step = 0
            previous_losses = []
            for j in range(len(train_data)-seq_length_in):
                start_time = time.time()

                encoder_inputs = train_data[j:(j + 1)].reshape(30, 1),
                decoder_outputs = train_label[j:(j + 1)]

                if transfer == 1:
                    source_data_1 = source_datas_1.iloc[current_step][1:].tolist()
                    source_data_1 = np.reshape(source_data_1, (30, 1))
                    source_data_2 = source_datas_2.iloc[current_step][1:].tolist()
                    source_data_2 = np.reshape(source_data_2, (30, 1))
                    source_data_3 = source_datas_3.iloc[current_step][1:].tolist()
                    source_data_3 = np.reshape(source_data_3, (30, 1))

                    _, step_loss, loss_summary, lr_summary = model.step(sess, encoder_inputs[0], decoder_outputs[0],
                                                                        source_data_1,
                                                                        source_data_2,
                                                                        source_data_3, False)
                else:
                    _, step_loss, loss_summary, lr_summary = model.step(sess, encoder_inputs[0], decoder_outputs[0],
                                                                        False)

                # if current_step % 100 == 0:
                #     print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss))

                current_step += 1

                # === step decay ===
                if current_step % learning_rate_step == 0:
                    sess.run(model.learning_rate_decay_op)

                previous_losses.append(step_loss)
            print('epoch: {0}, train loss: {1}'.format(epoch_index, sum(previous_losses)/len(previous_losses)))
            loss_list.append(sum(previous_losses)/len(previous_losses))
            epoch_index += 1
            # test(model, sess)
            # Save the model

            if (epoch_index == epoch) & (current_step % (len(train_data)-seq_length_in) == 0):
                print("current_step = ", current_step)
                #save loss log
                log_path = os.path.join('../../logs/', 'TL2-exp1-p1-epoch')
                # log_path = os.path.join('../../logs/', 'target0-exp1-epoch')
                data_frame = pd.DataFrame(data=loss_list)
                data_frame.to_csv(log_path, encoding='utf-8')

                test(model, sess, test_data, test_label, transfer)

                print("done in {0:.2f} ms".format((time.time() - start_time) * 1000))
                model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'model')),
                                 global_step=source_index)
                print("Saving the model...")

            # Reset global time and loss
            step_time, loss = 0, 0

            sys.stdout.flush()
        # model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'model')),
        #                  global_step=source_index)


def test(model, sess, test_data, test_label, transfer):
    seq_length_in = 30

    # save weight log
    weight_values = []
    for index_w in range(3):
        weight_value = model.w_h[index_w].eval()[0]
        print('mean value: ', sum(weight_value)/len(weight_value))
        weight_values.append(weight_value)
    print(weight_values)

    log_path = os.path.join('../../logs/weight/', 'TL2-exp1-p1-weight')
    data_frame = pd.DataFrame(data=weight_values)
    data_frame.to_csv(log_path, encoding='utf-8')

    file_full_path_1 = os.path.join('../../source_datas/', 'test_datas1')
    file_full_path_2 = os.path.join('../../source_datas/', 'test_datas2')
    file_full_path_3 = os.path.join('../../source_datas/', 'test_datas3')
    source_datas_1 = pd.read_csv(file_full_path_1)
    source_datas_2 = pd.read_csv(file_full_path_2)
    source_datas_3 = pd.read_csv(file_full_path_3)


    losses = []
    output_datas = []
    current_step = 0
    for j in range(len(test_label) - seq_length_in):

        encoder_inputs = test_data[j:(j + 1)].reshape(30, 1),
        decoder_outputs = test_label[j:(j + 1)]

        if transfer == 1:
            source_data_1 = source_datas_1.iloc[current_step][1:].tolist()
            source_data_1 = np.reshape(source_data_1, (30, 1))
            source_data_2 = source_datas_2.iloc[current_step][1:].tolist()
            source_data_2 = np.reshape(source_data_2, (30, 1))
            source_data_3 = source_datas_3.iloc[current_step][1:].tolist()
            source_data_3 = np.reshape(source_data_3, (30, 1))
            step_loss, loss_summary, output_data = model.step(sess, encoder_inputs[0], decoder_outputs[0],
                                                              source_data_1, source_data_2, source_data_3,
                                                              forward_only=True)
        else:
            step_loss, loss_summary, output_data = model.step(sess, encoder_inputs[0], decoder_outputs[0],
                                                     forward_only=True)

        current_step += 1
        output_datas.append(output_data[0][-1])
        losses.append(step_loss)
    mean_loss = sum(losses)/len(losses)
    # createData2.save_original_data('predict_outputs', output_datas)
    print('test loss: ', mean_loss)
    # plt.plot(output_datas)
    # plt.plot(test_label[0:len(test_label) - seq_length_in])
    # plt.legend(['pred', 'gt'])
    # plt.show()
    # createData2.draw_data(output_datas, len(output_datas), 'predict_outputs.pdf')


def tf_train(train_data_list, train_label_list, transfer, test_data, test_label, source_index=1):
    epoch = 150
    max_gradient_norm = 5
    batch_size = 128

    seq_length_in = 30
    seq_length_out = 1

    learning_rate = 0.05
    learning_rate_2 = 0.02
    learning_rate_decay_factor = 0.99
    learning_rate_step = 150

    num_layers = 1
    size = 1

    load = 0

    source_size = source_index

    with tf.Session() as sess:
        print("Creating %d layers." % (num_layers))

        model = create_model(sess, seq_length_in, seq_length_out, size, num_layers, max_gradient_norm,
                             batch_size, learning_rate, learning_rate_decay_factor, load, transfer, source_size)
        model.train_writer.add_graph(sess.graph)
        print("Model created")

        # === Training step ===
        # for j in range(2):
        for data_index in range(len(train_data_list)):
            if data_index < len(train_data_list)-1:
                sess.run(tf.assign(model.learning_rate, learning_rate))
            else:
                sess.run(tf.assign(model.learning_rate, learning_rate_2))
            sess.run(tf.assign(model.learning_rate, model.learning_rate * learning_rate_decay_factor))
            train_data = train_data_list[data_index]
            train_label = train_label_list[data_index]

            loss_list = []
            current_step = 0
            for epoch_index in range(epoch):
                current_step = 0
                previous_losses = []
                for j in range(len(train_data)-seq_length_in):
                    start_time = time.time()

                    encoder_inputs = train_data[j:(j + 1)].reshape(30, 1),
                    decoder_outputs = train_label[j:(j + 1)]

                    _, step_loss, loss_summary, lr_summary = model.step(sess, encoder_inputs[0], decoder_outputs[0], False)

                    current_step += 1

                    # === step decay ===
                    if current_step % learning_rate_step == 0:
                        sess.run(model.learning_rate_decay_op)


                    previous_losses.append(step_loss)
                print('epoch: {0}, train loss: {1}'.format(epoch_index, sum(previous_losses)/len(previous_losses)))
                loss_list.append(sum(previous_losses)/len(previous_losses))

                epoch_index += 1
            # test(model, sess)
            # Save the model
            if (epoch_index == epoch) & (current_step % (len(train_data)-seq_length_in) == 0):
                print("current_step = ", current_step)

                # save loss log
                log_path = os.path.join('../../logs/', 'pre-train-exp1-p2-epoch')
                data_frame = pd.DataFrame(data=loss_list)
                data_frame.to_csv(log_path, encoding='utf-8')

                test(model, sess, test_data, test_label, transfer)

                print("done in {0:.2f} ms".format((time.time() - start_time) * 1000))
                model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'model')),
                                 global_step=source_index)
                print("Saving the model...")

            # Reset global time and loss
            step_time, loss = 0, 0

            sys.stdout.flush()


if __name__ == '__main__':
    print('begin train')
    transfer = 1
    source_index = 3

    data_file_name = 'data_period150_w1_mu0_sigma002'
    # data_file_name = 'data_period100_w1_mu0_sigma002'
    # data_file_name = 'data_period200_w1_mu0_sigma002'
    # data_file_name = 'data_period20_w1_mu0_sigma002'
    # data_file_name = 'data_period120_w1_mu0_sigma001'
    # data_file_name = 'data_period150_w1_mu0_sigma03'

    datas = createData2.read_original_data(data_file_name)
    datas = createData2.normalization_data(datas)

    data_len = 450
    # data_len = len(datas)
    print('data len: ', data_len)
    data_train = datas[0:data_len]
    data_test = datas[data_len:]
    # data_test = datas[0:data_len]
    train_data, train_label = createData2.generate(data_train)

    # data_file_name2 = 'data_period100_w1_mu0_sigma002'
    # datas2 = createData2.read_original_data(data_file_name2)
    # datas2 = createData2.normalization_data(datas2)
    # data_len2 = len(datas2)
    # data_train2 = datas2[0:data_len2]
    # train_data2, train_label2 = createData2.generate(data_train2)
    #
    # data_file_name3 = 'data_period200_w1_mu0_sigma002'
    # datas3 = createData2.read_original_data(data_file_name3)
    # datas3 = createData2.normalization_data(datas3)
    # data_len3 = len(datas3)
    # data_train3 = datas3[0:data_len3]
    # train_data3, train_label3 = createData2.generate(data_train3)
    # #
    # data_file_name4 = 'data_period150_w1_mu0_sigma03'
    # datas4 = createData2.read_original_data(data_file_name4)
    # datas4 = createData2.normalization_data(datas4)
    # data_len4 = len(datas4)
    # data_train4 = datas4[0:data_len4]
    # train_data4, train_label4 = createData2.generate(data_train4)
    # #
    # train_data_list = []
    # train_label_list = []
    # train_data_list.append(train_data2)
    # train_data_list.append(train_data3)
    # train_data_list.append(train_data4)
    # train_data_list.append(train_data)
    # train_label_list.append(train_label2)
    # train_label_list.append(train_label3)
    # train_label_list.append(train_label4)
    # train_label_list.append(train_label)

    test_data, test_label = createData2.generate(data_test)

    print("train size: {0}".format(len(train_data)))

    # train_data = test_data
    # train_label = test_label

    # create_source_datas(train_data, train_label, transfer, test_data, test_label, source_index)

    train(train_data, train_label, transfer, test_data, test_label, source_index)
    # tf_train(train_data_list, train_label_list, transfer, test_data, test_label, source_index)

