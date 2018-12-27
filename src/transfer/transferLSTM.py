import tensorflow as tf
import os.path
import logging
import createData2


def load_model(model_name):
    print('load model')
    model_name = os.path.join('../../model/', model_name)
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(model_name + '.meta')
        new_saver.restore(sess, model_name)
        # variable_names = [v.name for v in tf.trainable_variables()]
        # values = sess.run(variable_names)
        # print(variable_names)
        # for k, v in zip(variable_names, values):
        #     print("Variable: ", k)
        #     print("Shape: ", v.shape)
        #     print(v)
        # print_num_of_total_parameters(output_detail=True, output_to_logging=True)

        # X_test, y_label = createData2.create_data(start=0, stop=15, num=500)
        # BATCH_SIZE = 128
        # TEST_EXAMPLES = X_test.shape[0]
        # test_losses = []
        #
        #
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


def create_source_lstm_state(session, path, input_state, ):
    print('get source lstm')


def create_target_lstm():
    print('create target lstm')


def print_num_of_total_parameters(output_detail=False, output_to_logging=False):
    total_parameters = 0
    parameters_string = ""

    for variable in tf.trainable_variables():

        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        if len(shape) == 1:
            parameters_string += ("%s %d, " % (variable.name, variable_parameters))
        else:
            parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))

    if output_to_logging:
        if output_detail:
            logging.info(parameters_string)
        logging.info("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
    else:
        if output_detail:
            print(parameters_string)
        print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))


def generate_transfer_model():
    print('generate transfer model')


if __name__ == '__main__':
    print('begin tansfer learning')
    load_model('test1.mode')
