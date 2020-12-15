import math
from config import *
import tf_slim as slim
import tensorflow as tf


def select_model(label_count, window_size_ms, window_stride_ms, dct_coefficient_count,
                 sample_rate=SAMPLE_RATE, clip_duration_ms=CLIP_DURATION_MS, name='dnn'):
    config = label_count, sample_rate, clip_duration_ms, window_size_ms, window_stride_ms, dct_coefficient_count
    if name == 'ds_cnn':
        return DS_CNN(config)


class KWSModel:
    def __init__(self, config):
        self.label_count = config[0]
        self.sample_rate = config[1]
        self.clip_duration_ms = config[2]
        self.window_size_ms = config[3]
        self.window_stride_ms = config[4]
        self.dct_coefficient_count = config[5]

        self.desired_samples = int(self.sample_rate * self.clip_duration_ms / 1000)
        self.window_size_samples = int(self.sample_rate * self.window_size_ms / 1000)
        self.window_stride_samples = int(self.sample_rate * self.window_stride_ms / 1000)
        self.length_minus_window = (self.desired_samples - self.window_size_samples)
        if self.length_minus_window < 0:
            self.spectrogram_length = 0
        else:
            self.spectrogram_length = 1 + int(self.length_minus_window / self.window_stride_samples)
        self.fingerprint_size = self.dct_coefficient_count * self.spectrogram_length

    def prepare_model_settings(self):
        return {'desired_samples': self.desired_samples,
                'window_size_samples': self.window_size_samples,
                'window_stride_samples': self.window_stride_samples,
                'spectrogram_length': self.spectrogram_length,
                'dct_coefficient_count': self.dct_coefficient_count,
                'fingerprint_size': self.fingerprint_size,
                'label_count': self.label_count,
                'sample_rate': self.sample_rate}

    def forward(self, fingerprint_input, model_size_info=None, is_training=True):
        raise NotImplementedError

    @staticmethod
    def load_variables_from_checkpoint(sess, start_checkpoint):
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        saver.restore(sess, start_checkpoint)


class DS_CNN(KWSModel):
    @staticmethod
    def depth_wise_separable_conv(inputs, num_pwc_filters, kernel_size, stride, sc):
        # skip point wise by setting num_outputs=None
        depth_wise_conv = slim.separable_convolution2d(inputs, num_outputs=None, stride=stride, depth_multiplier=1,
                                                       kernel_size=kernel_size, scope=sc + '/dw_conv')
        batch_norm = slim.batch_norm(depth_wise_conv, scope=sc + '/dw_conv/batch_norm')
        point_wise_conv = slim.convolution2d(batch_norm, num_pwc_filters, kernel_size=[1, 1], scope=sc + '/pw_conv')
        batch_norm = slim.batch_norm(point_wise_conv, scope=sc + '/pw_conv/batch_norm')
        return batch_norm

    def forward(self, fingerprint_input, model_size_info=None, is_training=True):
        dropout_prob = 1.0
        if is_training:
            dropout_prob = tf.compat.v1.placeholder(tf.float32, name='dropout_prob')
        label_count = self.label_count
        input_frequency_size = self.dct_coefficient_count
        input_time_size = self.spectrogram_length
        fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size, 1])

        t_dim = input_time_size
        f_dim = input_frequency_size

        # Extract model dimensions from model_size_info
        num_layers = model_size_info[0]
        conv_feat = [None] * num_layers
        conv_kt = [None] * num_layers
        conv_kf = [None] * num_layers
        conv_st = [None] * num_layers
        conv_sf = [None] * num_layers

        i = 1
        for layer_no in range(0, num_layers):
            conv_feat[layer_no] = model_size_info[i]
            i += 1
            conv_kt[layer_no] = model_size_info[i]
            i += 1
            conv_kf[layer_no] = model_size_info[i]
            i += 1
            conv_st[layer_no] = model_size_info[i]
            i += 1
            conv_sf[layer_no] = model_size_info[i]
            i += 1

        with tf.compat.v1.variable_scope('DS-CNN') as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                                activation_fn=None,
                                weights_initializer=slim.initializers.xavier_initializer(),
                                biases_initializer=slim.init_ops.zeros_initializer(),
                                outputs_collections=[end_points_collection]):
                with slim.arg_scope([slim.batch_norm],
                                    is_training=is_training,
                                    decay=0.96,
                                    updates_collections=None,
                                    activation_fn=tf.nn.relu):
                    for layer_no in range(0, num_layers):
                        if layer_no == 0:
                            net = slim.convolution2d(fingerprint_4d, conv_feat[layer_no],
                                                     [conv_kt[layer_no], conv_kf[layer_no]],
                                                     stride=[conv_st[layer_no], conv_sf[layer_no]],
                                                     padding='SAME', scope='conv_1')
                            net = slim.batch_norm(net, scope='conv_1/batch_norm')
                        else:
                            net = self.depth_wise_separable_conv(net, conv_feat[layer_no],
                                                                 kernel_size=[conv_kt[layer_no], conv_kf[layer_no]],
                                                                 stride=[conv_st[layer_no], conv_sf[layer_no]],
                                                                 sc='conv_ds_' + str(layer_no))
                        t_dim = math.ceil(t_dim / float(conv_st[layer_no]))
                        f_dim = math.ceil(f_dim / float(conv_sf[layer_no]))
                    net = slim.avg_pool2d(net, [t_dim, f_dim], scope='avg_pool')
            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            logits = tf.squeeze(slim.fully_connected(net, label_count, activation_fn=None, scope='fc1'))

        return logits, dropout_prob
