import random
import numpy as np
from utils import *
from config import *
from tqdm import tqdm
import tensorflow as tf
from data import AudioLoader
from models2 import select_model


def main(_):
    random.seed(RANDOM_SEED)
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    sess = tf.compat.v1.InteractiveSession()

    # model
    wanted_words = args.wanted_words.split(',')
    model = select_model(1, window_size_ms=args.window_size_ms,
                         window_stride_ms=args.window_stride_ms, dct_coefficient_count=args.dct_coefficient_count,
                         name=args.model_architecture)
    model_settings = model.prepare_model_settings()
    print('-----\nModel settings: {}'.format(model_settings))

    # data
    audio_loader = AudioLoader(args.data_dir, wanted_words, SILENCE_PERCENTAGE, VOCAL_PERCENTAGE, NEGATIVE_PERCENTAGE,
                               VALIDATION_PERCENTAGE, TESTING_PERCENTAGE, model_settings)

    fingerprint_size = model_settings['fingerprint_size']
    time_shift_samples = int((TIME_SHIFT_MS * model_settings['sample_rate']) / 1000)
    training_steps_list = list(map(int, args.training_steps.split(',')))
    learning_rates_list = list(map(float, args.learning_rate.split(',')))

    fingerprint_input = tf.compat.v1.placeholder(tf.float32, [None, fingerprint_size], name='fingerprint_input')
    ground_truth_input = tf.compat.v1.placeholder(tf.float32, [None], name='groundtruth_input')
    logits, dropout_prob = model.forward(fingerprint_input, args.model_size_info)
    logits = tf.sigmoid(logits)

    # Create the back propagation and training evaluation machinery in the graph.
    with tf.name_scope('mae'):
        mae_loss = tf.reduce_mean(tf.compat.v1.losses.absolute_difference(ground_truth_input, logits))
    tf.compat.v1.summary.scalar('mae', mae_loss)

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.name_scope('train'), tf.control_dependencies(update_ops):
        learning_rate_input = tf.compat.v1.placeholder(tf.float32, [], name='learning_rate_input')
        train_step = tf.compat.v1.train.AdamOptimizer(learning_rate_input).minimize(mae_loss)

    evaluation_step = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(ground_truth_input, logits))))
    tf.compat.v1.summary.scalar('accuracy', evaluation_step)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    increment_global_step = tf.compat.v1.assign(global_step, global_step + 1)
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged_summaries = tf.compat.v1.summary.merge_all()
    train_writer = tf.compat.v1.summary.FileWriter(os.path.join(args.summaries_dir, 'train'), sess.graph)
    validation_writer = tf.compat.v1.summary.FileWriter(os.path.join(args.summaries_dir, 'val'))

    tf.compat.v1.global_variables_initializer().run()
    params = tf.compat.v1.trainable_variables()
    num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
    print('Total number of Parameters: {}\n-----'.format(num_params))

    # Save graph.pbtxt.
    tf.io.write_graph(sess.graph_def, args.train_dir, '{}.pbtxt'.format(args.model_architecture))

    # training loop
    step = 0
    epoch = 0
    best_accuracy = 100
    training_steps_max = np.sum(training_steps_list)
    train_size = audio_loader.size('training')
    while step < training_steps_max + 1:
        epoch += 1
        audio_loader.shuffle(set_index='training')
        for offset in range(0, train_size, args.batch_size):
            step += 1
            if step >= training_steps_max + 1:
                break
            training_steps_sum = 0
            learning_rate_value = 0
            for i in range(len(training_steps_list)):
                training_steps_sum += training_steps_list[i]
                if step <= training_steps_sum:
                    learning_rate_value = learning_rates_list[i]
                    break

            # train
            train_fingerprints, train_ground_truth = audio_loader \
                .load_batch(sess, args.batch_size, offset,
                            BACKGROUND_FREQUENCY, BACKGROUND_VOLUME,
                            BACKGROUND_SILENCE_FREQUENCY, BACKGROUND_SILENCE_VOLUME,
                            DOWN_VOLUME_FREQUENCY, DOWN_VOLUME_RANGE,
                            time_shift_samples, mode='training')
            train_summary, train_accuracy, loss, _, _ = sess.run(
                [merged_summaries, evaluation_step, mae_loss, train_step, increment_global_step],
                feed_dict={fingerprint_input: train_fingerprints,
                           ground_truth_input: train_ground_truth,
                           learning_rate_input: learning_rate_value,
                           dropout_prob: 1.0})
            train_writer.add_summary(train_summary, step)
            tf.compat.v1.logging.info('Epoch {} - Step {}: train accuracy {}, loss {}, lr {}'.format(
                epoch, step, train_accuracy * 100, loss, learning_rate_value))

            # val
            if step % args.eval_step_interval == 0:
                total_accuracy = 0
                val_size = audio_loader.size('validation')
                for i in tqdm(range(0, val_size, args.batch_size)):
                    val_fingerprints, val_ground_truth = audio_loader \
                        .load_batch(sess, args.batch_size, offset=i, background_frequency=0,
                                    background_volume_range=0, time_shift=0, mode='validation')
                    val_summary, val_accuracy = sess.run([merged_summaries, evaluation_step],
                                                         feed_dict={fingerprint_input: val_fingerprints,
                                                                    ground_truth_input: val_ground_truth,
                                                                    dropout_prob: 1.0})
                    validation_writer.add_summary(val_summary, step)
                    batch_size = min(args.batch_size, val_size - i)
                    total_accuracy += (val_accuracy * batch_size) / val_size
                tf.compat.v1.logging.info('Step {}: val accuracy {}'.format(step, total_accuracy))

                # Save the model checkpoint when validation accuracy improves
                if total_accuracy <= best_accuracy:
                    best_accuracy = total_accuracy
                    checkpoint_path = os.path.join(
                        args.train_dir, 'best',
                        '{}_{}.ckpt'.format(args.model_architecture, str(int(best_accuracy * 10000))))
                    saver.save(sess, checkpoint_path, global_step=step)
                    tf.compat.v1.logging.info('Saving best model to {} - step {}'.format(checkpoint_path, step))
                tf.compat.v1.logging.info('So far the best validation accuracy is %.2f%%' % (best_accuracy * 100))

    # test
    print('Testing')
    test_size = audio_loader.size(mode='testing')
    tf.compat.v1.logging.info('set_size=%d', test_size)
    total_accuracy = 0
    for i in tqdm(range(0, test_size, args.batch_size)):
        test_fingerprints, test_ground_truth = audio_loader \
            .load_batch(sess, args.batch_size, offset=i, background_frequency=0,
                        background_volume_range=0, time_shift=0, mode='testing')
        test_summary, test_accuracy = sess.run([merged_summaries, evaluation_step],
                                               feed_dict={
                                                   fingerprint_input: test_fingerprints,
                                                   ground_truth_input: test_ground_truth,
                                                   dropout_prob: 1.0})
        batch_size = min(args.batch_size, test_size - i)
        total_accuracy += (test_accuracy * batch_size) / test_size
    tf.compat.v1.logging.info('Final accuracy {}'.format(total_accuracy))

    # close
    sess.close()


if __name__ == '__main__':
    args = prepare_normal_config()
    tf.compat.v1.app.run(main=main)
