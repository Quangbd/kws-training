import math
import tf_slim as slim
from utils import *


def select_model(window_size_ms, window_stride_ms, dct_coefficient_count,
                 sample_rate=SAMPLE_RATE, clip_duration_ms=CLIP_DURATION_MS, name='dnn'):
    config = 2, sample_rate, clip_duration_ms, window_size_ms, window_stride_ms, dct_coefficient_count
    if name == 'single_fc':
        return SingleFC(config)
    elif name == 'dnn':
        return DNN(config)
    elif name == 'conv':
        return CONV(config)
    elif name == 'cnn':
        return CNN(config)
    elif name == 'lstm':
        return LSTM(config)
    elif name == 'ds_cnn':
        return DS_CNN(config)
    elif name == 'crnn':
        return CRNN(config)


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


class SingleFC(KWSModel):
    """
    (fingerprint_input)
                v
        [MatMul]<-(weights)
                v
        [BiasAdd]<-(bias)
                v
    """

    def forward(self, fingerprint_input, model_size_info=None, is_training=True):
        dropout_prob = None
        if is_training:
            dropout_prob = tf.compat.v1.placeholder(tf.float32, name='dropout_prob')
        weights = tf.Variable(tf.random.truncated_normal([self.fingerprint_size, self.label_count], stddev=0.001))
        bias = tf.Variable(tf.zeros([self.label_count]))
        logits = tf.matmul(fingerprint_input, weights) + bias
        return logits, dropout_prob


class DNN(KWSModel):
    def forward(self, fingerprint_input, model_size_info=None, is_training=True):
        dropout_prob = 0
        if is_training:
            dropout_prob = tf.compat.v1.placeholder(tf.float32, name='dropout_prob')
        num_layers = len(model_size_info)
        layer_dim = [self.fingerprint_size]
        layer_dim.extend(model_size_info)
        flow = fingerprint_input
        tf.summary.histogram('input', flow)
        for i in range(1, num_layers + 1):
            with tf.compat.v1.variable_scope('fc' + str(i)):
                weights = tf.compat.v1.get_variable('W', shape=[layer_dim[i - 1], layer_dim[i]],
                                                    initializer=tf.initializers.glorot_uniform())
                tf.summary.histogram('fc_' + str(i) + '_w', weights)
                bias = tf.compat.v1.get_variable('b', shape=[layer_dim[i]])
                tf.summary.histogram('fc_' + str(i) + '_b', bias)
                flow = tf.matmul(flow, weights) + bias
                flow = tf.nn.relu(flow)
                if is_training:
                    flow = tf.nn.dropout(flow, 1 - dropout_prob)
        t_weights = tf.compat.v1.get_variable('final_fc', shape=[layer_dim[-1], self.label_count],
                                              initializer=tf.initializers.glorot_uniform())
        t_bias = tf.Variable(tf.zeros([self.label_count]))
        logits = tf.matmul(flow, t_weights) + t_bias
        return logits, dropout_prob


class CONV(KWSModel):
    """
    (fingerprint_input)
            v
        [Conv2D]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
        [Relu]
            v
        [MaxPool]
            v
        [Conv2D]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
        [Relu]
            v
        [MaxPool]
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
    """

    def forward(self, fingerprint_input, model_size_info=None, is_training=True):
        dropout_prob = 1.0
        if is_training:
            dropout_prob = tf.compat.v1.placeholder(tf.float32, name='dropout_prob')
        input_frequency_size = self.dct_coefficient_count
        input_time_size = self.spectrogram_length
        fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size, 1])

        # first---
        first_filter_width = 8
        first_filter_height = 20
        first_filter_count = 64
        first_weights = tf.Variable(tf.random.truncated_normal(
            [first_filter_height, first_filter_width, 1, first_filter_count], stddev=0.01))
        first_bias = tf.Variable(tf.zeros([first_filter_count]))

        first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                                  'SAME') + first_bias
        first_relu = tf.nn.relu(first_conv)
        if is_training:
            first_dropout = tf.nn.dropout(first_relu, 1 - dropout_prob)
        else:
            first_dropout = first_relu
        max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        # second---
        second_filter_width = 4
        second_filter_height = 10
        second_filter_count = 64
        second_weights = tf.Variable(tf.random.truncated_normal(
            [second_filter_height, second_filter_width, first_filter_count, second_filter_count], stddev=0.01))
        second_bias = tf.Variable(tf.zeros([second_filter_count]))
        second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1], 'SAME') + second_bias
        second_relu = tf.nn.relu(second_conv)
        if is_training:
            second_dropout = tf.nn.dropout(second_relu, 1 - dropout_prob)
        else:
            second_dropout = second_relu

        # ---
        second_conv_shape = second_dropout.get_shape()
        second_conv_output_width = second_conv_shape[2]
        second_conv_output_height = second_conv_shape[1]
        second_conv_element_count = int(second_conv_output_width * second_conv_output_height * second_filter_count)
        flattened_second_conv = tf.reshape(second_dropout, [-1, second_conv_element_count])
        final_fc_weights = tf.Variable(tf.random.truncated_normal(
            [second_conv_element_count, self.label_count], stddev=0.01))
        final_fc_bias = tf.Variable(tf.zeros([self.label_count]))
        final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias

        return final_fc, dropout_prob


class CNN(KWSModel):
    def forward(self, fingerprint_input, model_size_info=None, is_training=True):
        dropout_prob = 1.0
        if is_training:
            dropout_prob = tf.compat.v1.placeholder(tf.float32, name='dropout_prob')
        input_frequency_size = self.dct_coefficient_count
        input_time_size = self.spectrogram_length
        fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size, 1])

        first_filter_count = model_size_info[0]
        first_filter_height = model_size_info[1]  # time axis
        first_filter_width = model_size_info[2]  # frequency axis
        first_filter_stride_y = model_size_info[3]  # time axis
        first_filter_stride_x = model_size_info[4]  # frequency_axis

        second_filter_count = model_size_info[5]
        second_filter_height = model_size_info[6]  # time axis
        second_filter_width = model_size_info[7]  # frequency axis
        second_filter_stride_y = model_size_info[8]  # time axis
        second_filter_stride_x = model_size_info[9]  # frequency_axis

        linear_layer_size = model_size_info[10]
        fc_size = model_size_info[11]

        # first conv
        first_weights = tf.Variable(tf.random.truncated_normal(
            [first_filter_height, first_filter_width, 1, first_filter_count], stddev=0.01))
        first_bias = tf.Variable(tf.zeros([first_filter_count]))
        first_conv = tf.nn.conv2d(fingerprint_4d, first_weights,
                                  [1, first_filter_stride_y, first_filter_stride_x, 1], 'VALID') + first_bias
        first_conv = tf.compat.v1.layers.batch_normalization(first_conv, training=is_training, name='bn1')
        first_relu = tf.nn.relu(first_conv)
        if is_training:
            first_dropout = tf.nn.dropout(first_relu, 1 - dropout_prob)
        else:
            first_dropout = first_relu
        first_conv_output_width = math.ceil((input_frequency_size - first_filter_width + 1) / first_filter_stride_x)
        first_conv_output_height = math.ceil((input_time_size - first_filter_height + 1) / first_filter_stride_y)

        # second conv
        second_weights = tf.Variable(tf.random.truncated_normal(
            [second_filter_height, second_filter_width, first_filter_count, second_filter_count], stddev=0.01))
        second_bias = tf.Variable(tf.zeros([second_filter_count]))
        second_conv = tf.nn.conv2d(first_dropout, second_weights,
                                   [1, second_filter_stride_y, second_filter_stride_x, 1], 'VALID') + second_bias
        second_conv = tf.compat.v1.layers.batch_normalization(second_conv, training=is_training, name='bn2')
        second_relu = tf.nn.relu(second_conv)
        if is_training:
            second_dropout = tf.nn.dropout(second_relu, 1 - dropout_prob)
        else:
            second_dropout = second_relu
        second_conv_output_width = math.ceil((first_conv_output_width - second_filter_width + 1)
                                             / second_filter_stride_x)
        second_conv_output_height = math.ceil((first_conv_output_height - second_filter_height + 1)
                                              / second_filter_stride_y)

        second_conv_element_count = int(second_conv_output_width * second_conv_output_height * second_filter_count)
        flattened_second_conv = tf.reshape(second_dropout, [-1, second_conv_element_count])

        # linear_layer
        t_weights = tf.compat.v1.get_variable('W', shape=[second_conv_element_count, linear_layer_size],
                                              initializer=tf.initializers.glorot_uniform())
        t_bias = tf.compat.v1.get_variable('b', shape=[linear_layer_size])
        flow = tf.matmul(flattened_second_conv, t_weights) + t_bias

        # first_fc
        first_fc_output_channels = fc_size
        first_fc_weights = tf.Variable(tf.random.truncated_normal(
            [linear_layer_size, first_fc_output_channels], stddev=0.01))
        first_fc_bias = tf.Variable(tf.zeros(first_fc_output_channels))
        first_fc = tf.matmul(flow, first_fc_weights) + first_fc_bias
        first_fc = tf.compat.v1.layers.batch_normalization(first_fc, training=is_training, name='bn3')
        first_fc = tf.nn.relu(first_fc)
        if is_training:
            final_fc_input = tf.nn.dropout(first_fc, 1 - dropout_prob)
        else:
            final_fc_input = first_fc
        label_count = self.label_count

        # final_fc
        final_fc_weights = tf.Variable(tf.random.truncated_normal(
            [first_fc_output_channels, label_count], stddev=0.01))
        final_fc_bias = tf.Variable(tf.zeros([label_count]))
        final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
        return final_fc, dropout_prob


class LSTM(KWSModel):
    def forward(self, fingerprint_input, model_size_info=None, is_training=True):
        dropout_prob = 1.0
        if is_training:
            dropout_prob = tf.compat.v1.placeholder(tf.float32, name='dropout_prob')
        input_frequency_size = self.dct_coefficient_count
        input_time_size = self.spectrogram_length
        fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size])
        num_classes = self.label_count
        projection_units = model_size_info[0]
        lstm_units = model_size_info[1]

        with tf.name_scope('LSTM-Layer'):
            with tf.compat.v1.variable_scope('lstm'):
                lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(lstm_units,
                                                              use_peepholes=True, num_proj=projection_units)
                _, last = tf.compat.v1.nn.dynamic_rnn(cell=lstm_cell, inputs=fingerprint_4d, dtype=tf.float32)
                flow = last[-1]

        with tf.name_scope('Output-Layer'):
            weights_o = tf.compat.v1.get_variable('W_o', shape=[projection_units, num_classes],
                                                  initializer=tf.initializers.glorot_uniform())
            bias_o = tf.compat.v1.get_variable('b_o', shape=[num_classes])
            logits = tf.matmul(flow, weights_o) + bias_o
        return logits, dropout_prob


class CRNN(KWSModel):
    def forward(self, fingerprint_input, model_size_info=None, is_training=True):
        dropout_prob = 1.0
        if is_training:
            dropout_prob = tf.compat.v1.placeholder(tf.float32, name='dropout_prob')
        input_frequency_size = self.dct_coefficient_count
        input_time_size = self.spectrogram_length
        fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size, 1])

        # CNN part
        first_filter_count = model_size_info[0]
        first_filter_height = model_size_info[1]
        first_filter_width = model_size_info[2]
        first_filter_stride_y = model_size_info[3]
        first_filter_stride_x = model_size_info[4]

        first_weights = tf.compat.v1.get_variable('W', shape=[first_filter_height, first_filter_width,
                                                              1, first_filter_count],
                                                  initializer=tf.initializers.glorot_uniform())
        first_bias = tf.Variable(tf.zeros([first_filter_count]))
        first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, first_filter_stride_y, first_filter_stride_x, 1],
                                  'VALID') + first_bias
        first_relu = tf.nn.relu(first_conv)
        if is_training:
            first_dropout = tf.nn.dropout(first_relu, 1 - dropout_prob)
        else:
            first_dropout = first_relu
        first_conv_output_width = int(
            math.floor((input_frequency_size - first_filter_width + first_filter_stride_x) / first_filter_stride_x))
        first_conv_output_height = int(
            math.floor((input_time_size - first_filter_height + first_filter_stride_y) / first_filter_stride_y))

        # GRU part
        num_rnn_layers = model_size_info[5]
        rnn_units = model_size_info[6]
        flow = tf.reshape(first_dropout, [-1, first_conv_output_height, first_conv_output_width * first_filter_count])
        cell_fw = []
        cell_bw = []
        for i in range(num_rnn_layers):
            cell_fw.append(tf.compat.v1.nn.rnn_cell.GRUCell(rnn_units))
        cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cell_fw)
        _, last = tf.compat.v1.nn.dynamic_rnn(cell=cells, inputs=flow, dtype=tf.float32)
        flow_dim = rnn_units
        flow = last[-1]

        first_fc_output_channels = model_size_info[7]
        first_fc_weights = tf.compat.v1.get_variable('fcw', shape=[flow_dim, first_fc_output_channels],
                                                     initializer=tf.initializers.glorot_uniform())

        first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
        first_fc = tf.nn.relu(tf.matmul(flow, first_fc_weights) + first_fc_bias)
        if is_training:
            final_fc_input = tf.nn.dropout(first_fc, 1 - dropout_prob)
        else:
            final_fc_input = first_fc

        label_count = self.label_count
        final_fc_weights = tf.Variable(tf.random.truncated_normal([first_fc_output_channels, label_count], stddev=0.01))

        final_fc_bias = tf.Variable(tf.zeros([label_count]))
        final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
        return final_fc, dropout_prob


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
            logits = slim.fully_connected(net, label_count, activation_fn=None, scope='fc1')
        return logits, dropout_prob
