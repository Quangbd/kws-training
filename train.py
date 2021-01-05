import random
import numpy as np
from utils import *
from constant import *
from tqdm import tqdm
import tensorflow as tf
from data import AudioLoader
from models import select_model


def init_session():
    random.seed(RANDOM_SEED)
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    sess = tf.compat.v1.InteractiveSession()
    return sess


def init_model():
    model = select_model(window_size_ms=args.window_size_ms, window_stride_ms=args.window_stride_ms,
                         dct_coefficient_count=args.dct_coefficient_count,
                         name=args.model_architecture)
    model_settings = model.prepare_model_settings()
    print('-----\nModel settings: {}'.format(model_settings))
    return model, model_settings


def init_data(model_settings):
    return AudioLoader(args.data_dir, args.silence_percentage, args.negative_percentage,
                       args.validation_percentage, args.testing_percentage,
                       model_settings, augment_dir=args.augment_dir)


def init_placeholder(model_settings, is_train=True):
    time_shift_samples = None
    training_steps_list = None
    learning_rates_list = None
    fingerprint_size = model_settings['fingerprint_size']
    if is_train:
        time_shift_samples = int((TIME_SHIFT_MS * model_settings['sample_rate']) / 1000)
        training_steps_list = list(map(int, args.training_steps.split(',')))
        learning_rates_list = list(map(float, args.learning_rate.split(',')))
    fingerprint_input = tf.compat.v1.placeholder(tf.float32, [None, fingerprint_size], name='fingerprint_input')
    ground_truth_input = tf.compat.v1.placeholder(tf.float32, [None, model_settings['label_count']],
                                                  name='groundtruth_input')
    return time_shift_samples, training_steps_list, learning_rates_list, fingerprint_input, ground_truth_input


def init_graph(model, model_settings, fingerprint_input, ground_truth_input, is_train=True):
    logits, dropout_prob = model.forward(fingerprint_input, args.model_size_info)

    if is_train:
        # Create the back propagation and training evaluation machinery in the graph.
        with tf.name_scope('cross_entropy'):
            cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=ground_truth_input, logits=logits))
        tf.compat.v1.summary.scalar('cross_entropy', cross_entropy_mean)

        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.name_scope('train'), tf.control_dependencies(update_ops):
            learning_rate_input = tf.compat.v1.placeholder(tf.float32, [], name='learning_rate_input')
            train_step = tf.compat.v1.train.AdamOptimizer(learning_rate_input).minimize(cross_entropy_mean)

    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    confusion_matrix = tf.math.confusion_matrix(expected_indices, predicted_indices,
                                                num_classes=model_settings['label_count'])
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.compat.v1.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, cross_entropy_mean, train_step, learning_rate_input, confusion_matrix, dropout_prob


def main(_):
    sess = init_session()
    model, model_settings = init_model()
    audio_loader = init_data(model_settings)
    time_shift_samples, training_steps_list, learning_rates_list, fingerprint_input, ground_truth_input = \
        init_placeholder(model_settings)
    evaluation_step, cross_entropy_mean, train_step, learning_rate_input, confusion_matrix, dropout_prob = \
        init_graph(model, model_settings, fingerprint_input, ground_truth_input)

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
    best_accuracy = 0
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
                .load_batch(sess, args.batch_size, offset=offset,
                            background_frequency=args.background_frequency,
                            background_silence_frequency=args.background_silence_frequency,
                            background_silence_volume_range=args.background_silence_volume,
                            time_shift=time_shift_samples, mode='training')
            train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
                [merged_summaries, evaluation_step, cross_entropy_mean, train_step, increment_global_step],
                feed_dict={fingerprint_input: train_fingerprints,
                           ground_truth_input: train_ground_truth,
                           learning_rate_input: learning_rate_value,
                           dropout_prob: 1.0})
            train_writer.add_summary(train_summary, step)
            tf.compat.v1.logging.info('Epoch {} - Step {}: train accuracy {}, cross entropy {}, lr {}'.format(
                epoch, step, train_accuracy * 100, cross_entropy_value, learning_rate_value))

            # val
            if step % args.eval_step_interval == 0:
                total_accuracy = 0
                val_size = audio_loader.size('validation')
                total_conf_matrix = None
                for i in tqdm(range(0, val_size, args.batch_size)):
                    val_fingerprints, val_ground_truth = audio_loader \
                        .load_batch(sess, args.batch_size, offset=i, background_frequency=0,
                                    time_shift=0, mode='validation')
                    val_summary, val_accuracy, val_matrix = sess.run(
                        [merged_summaries, evaluation_step, confusion_matrix],
                        feed_dict={
                            fingerprint_input: val_fingerprints,
                            ground_truth_input: val_ground_truth,
                            dropout_prob: 1.0})
                    validation_writer.add_summary(val_summary, step)
                    batch_size = min(args.batch_size, val_size - i)
                    total_accuracy += (val_accuracy * batch_size) / val_size
                    if total_conf_matrix is None:
                        total_conf_matrix = val_matrix
                    else:
                        total_conf_matrix += val_matrix
                tf.compat.v1.logging.info('Confusion matrix: \n %s' % total_conf_matrix)
                tf.compat.v1.logging.info('Step {}: val accuracy {}'.format(step, total_accuracy))

                # Save the model checkpoint when validation accuracy improves
                if total_accuracy >= best_accuracy:
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
    total_conf_matrix = None
    for i in tqdm(range(0, test_size, args.batch_size)):
        test_fingerprints, test_ground_truth = audio_loader \
            .load_batch(sess, args.batch_size, offset=i, background_frequency=0,
                        time_shift=0, mode='testing')
        test_summary, test_accuracy, test_matrix = sess.run(
            [merged_summaries, evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: test_fingerprints,
                ground_truth_input: test_ground_truth,
                dropout_prob: 1.0})
        batch_size = min(args.batch_size, test_size - i)
        total_accuracy += (test_accuracy * batch_size) / test_size
        if total_conf_matrix is None:
            total_conf_matrix = test_matrix
        else:
            total_conf_matrix += test_matrix
    tf.compat.v1.logging.info('Final confusion matrix: \n %s' % total_conf_matrix)
    tf.compat.v1.logging.info('Final accuracy {}'.format(total_accuracy))

    # close
    sess.close()


if __name__ == '__main__':
    args = prepare_config()
    tf.compat.v1.app.run(main=main)
