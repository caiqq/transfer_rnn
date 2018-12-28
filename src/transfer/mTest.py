import tensorflow as tf
import os
import time
import sys

import createData2
import SimpleModel
import TransferSimpleModel

train_dir = os.path.normpath(os.path.join('../../model/', 'train/'))


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


def train():
    transfer = 1

    train_data, train_label = createData2.create_data(start=0, stop=15, num=500, w=1)
    seq_length_in = 30
    seq_length_out = 1
    max_gradient_norm = 5
    batch_size = 128
    learning_rate = 0.05
    learning_rate_decay_factor = 0.95
    num_layers = 1
    size = 1
    learning_rate_step = 128
    save_every = 128

    load = 0

    iterations = 400

    source_size = 1

    with tf.Session() as sess:
        print("Creating %d layers of %d units." % (num_layers, size))

        model = create_model(sess, seq_length_in, seq_length_out, size, num_layers, max_gradient_norm,
                             batch_size, learning_rate, learning_rate_decay_factor, load, transfer, source_size)
        model.train_writer.add_graph(sess.graph)
        print("Model created")

        step_time, loss, val_loss = 0.0, 0.0, 0.0
        current_step = 0 if load <= 0 else load + 1
        previous_losses = []

        step_time, loss = 0, 0

        for j in range(iterations):

            start_time = time.time()

            # === Training step ===
            encoder_inputs = train_data[j:(j + 1)].reshape(30, 1),
            # print('encoder_input: {0}' .format(encoder_inputs[0]))
            # print(encoder_inputs[0].shape)
            decoder_outputs = train_label[j:(j + 1)]
            # print('decoder_outputs: {0}'.format(decoder_outputs[0]))
            # print(decoder_outputs.shape)

            _, step_loss, loss_summary, lr_summary = model.step(sess, encoder_inputs[0], decoder_outputs[0], False)

            model.train_writer.add_summary(loss_summary, current_step)
            model.train_writer.add_summary(lr_summary, current_step)

            if current_step % 50 == 0:
                print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss))

            step_time += (time.time() - start_time) / size
            loss += step_loss / size
            current_step += 1

            # === step decay ===
            if current_step % learning_rate_step == 0:
                sess.run(model.learning_rate_decay_op)

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % batch_size == 0:

                # === Validation with randomly chosen seeds ===
                forward_only = True

                step_loss, loss_summary = model.step(sess, encoder_inputs[0], decoder_outputs[0], forward_only)
                val_loss = step_loss  # Loss book-keeping

                model.test_writer.add_summary(loss_summary, current_step)

                print()
                print("{0: <16} |".format("milliseconds"), end="")

                print()
                print("============================\n"
                      "Global step:         %d\n"
                      "Learning rate:       %.4f\n"
                      "Step-time (ms):     %.4f\n"
                      "Train loss avg:      %.4f\n"
                      "--------------------------\n"
                      "Val loss:            %.4f\n"
                      "============================" % (model.global_step.eval(),
                                                        model.learning_rate.eval(),
                                                        step_time * 1000, loss,
                                                        val_loss))
                print()

                previous_losses.append(loss)

                # Save the model
                source_index = 1
                if current_step % save_every == 0:
                    print("Saving the model...");
                    start_time = time.time()
                    model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'model')),
                                     global_step=source_index)
                    print("done in {0:.2f} ms".format((time.time() - start_time) * 1000))

                # Reset global time and loss
                step_time, loss = 0, 0
                current_step = current_step +1
                sys.stdout.flush()


if __name__ == '__main__':
    print('begin train')
    train()
