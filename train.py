import os
import wandb
import numpy as np
from utils import *
from tqdm import tqdm
from constant import *
import tensorflow as tf
from data import AudioLoader
import tensorflow_addons as tfa


def micro_accuracy(confusion_matrix):
    if sum(confusion_matrix[1]) == 0:
        return confusion_matrix[0][0] / sum(confusion_matrix[0]), 1
    # Negative, positive
    return confusion_matrix[0][0] / sum(confusion_matrix[0]), confusion_matrix[1][1] / sum(confusion_matrix[1])


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


def main(_):
    sess = init_session()
    model, model_settings = init_model(args)
    print('-----\nModel settings: {}'.format(model_settings))
    audio_loader = AudioLoader(args.data_dir, args.silence_percentage, args.negative_percentage,
                               args.validation_percentage, model_settings, augment_dir=args.augment_dir)
    time_shift_samples, training_steps_list, learning_rates_list, fingerprint_input, ground_truth_input = \
        init_placeholder(model_settings)
    w_config = {'architecture': args.model_architecture,
                'background_frequency': args.background_frequency,
                'background_silence_frequency': args.background_silence_frequency,
                'background_silence_volume': args.background_silence_volume,
                'silence_percentage': args.silence_percentage,
                'learning_rate': learning_rates_list[0],
                'loss_method': args.loss_method,
                'batch_size': args.batch_size}
    w_config.update(audio_loader.total_sample_count)
    wandb.init(project='kws', name=args.name, config=w_config)

    # init graph
    logits, dropout_prob = model.forward(fingerprint_input, args.model_size_info)

    # Create the back propagation and training evaluation machinery in the graph.
    with tf.name_scope('cross_entropy'):
        if args.loss_method == 'fe':  # Focal entropy loss
            cross_entropy_mean = tf.reduce_mean(tfa.losses.sigmoid_focal_crossentropy(
                y_true=ground_truth_input, y_pred=tf.nn.softmax(logits)))
        else:  # Cross entropy loss
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
            train_fingerprints, train_ground_truth, positive_count, negative_count = audio_loader \
                .load_batch(sess, args.batch_size, offset=offset,
                            background_frequency=args.background_frequency,
                            background_silence_frequency=args.background_silence_frequency,
                            background_silence_volume_range=args.background_silence_volume,
                            time_shift=time_shift_samples, mode='training')
            train_matrix, train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
                [confusion_matrix,
                 merged_summaries, evaluation_step, cross_entropy_mean, train_step, increment_global_step],
                feed_dict={fingerprint_input: train_fingerprints,
                           ground_truth_input: train_ground_truth,
                           learning_rate_input: learning_rate_value,
                           dropout_prob: 1.0})
            train_writer.add_summary(train_summary, step)
            train_negative_acc, train_positive_acc = micro_accuracy(train_matrix)
            wandb.log({'train_epoch': epoch, 'train_acc': train_accuracy, 'train_negative_acc': train_negative_acc,
                       'train_positive_acc': train_positive_acc, 'train_loss': cross_entropy_value})
            print('Epoch {} - Step {}: train acc {}, nea {} poa {}, cross entropy {}, lr {}, positive {}, negative {}'
                  .format(epoch, step, round(train_accuracy * 100, 2),
                          round(train_negative_acc * 100, 2), round(train_positive_acc * 100, 2),
                          cross_entropy_value, learning_rate_value, positive_count, negative_count))

            # val
            if step % args.eval_step_interval == 0:
                total_accuracy = 0
                val_size = audio_loader.size('validation')
                total_conf_matrix = None
                for i in tqdm(range(0, val_size, args.batch_size)):
                    val_fingerprints, val_ground_truth, _, _ = audio_loader \
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
                print('Confusion matrix: \n %s' % total_conf_matrix)
                val_negative_acc, val_positive_acc = micro_accuracy(train_matrix)
                wandb.log({'val_acc': total_accuracy, 'val_negative_acc': val_negative_acc,
                           'val_positive_acc': val_positive_acc})
                print('Step {}: val accuracy {}, negative acc {} - positive acc {}'
                      .format(step, total_accuracy, val_negative_acc, val_positive_acc))

                # Save the model checkpoint when validation accuracy improves
                if total_accuracy >= best_accuracy:
                    best_micro_accuracy = total_accuracy
                    checkpoint_path = os.path.join(
                        args.train_dir, 'best',
                        '{}_{}.ckpt'.format(args.model_architecture, str(int(best_micro_accuracy * 10000))))
                    saver.save(sess, checkpoint_path, global_step=step)
                    print('Saving best model to {} - step {}'.format(checkpoint_path, step))
                print('So far the best validation accuracy is %.2f%%' % (best_accuracy * 100))

    # close
    wandb.finish()
    sess.close()


if __name__ == '__main__':
    args = prepare_config()
    tf.compat.v1.app.run(main=main)
