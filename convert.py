import os
from utils import *
from constant import *
import tensorflow as tf
from test import load_model
from train import init_session
from tensorflow.python.framework import graph_util


def checkpoint2pb():
    sess = init_session()
    load_model(args, ModelType.CHECKPOINT, sess)
    nodes = [op.name for op in tf.compat.v1.get_default_graph().get_operations()]
    print(nodes)
    # Turn all the variables into inline constants inside the graph and save it.
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ['labels_sigmoid'])
    tf.compat.v1.train.write_graph(frozen_graph_def,
                                   os.path.dirname(args.pb),
                                   os.path.basename(args.pb),
                                   as_text=False)
    tf.compat.v1.logging.info('Saved frozen graph to %s', args.pb)
    sess.close()


def pb2tflite():
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        args.pb,
        input_arrays=['decoded_sample_data'],
        output_arrays=['labels_sigmoid'])
    converter.allow_custom_ops = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(args.tflite, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    args = prepare_config()
    checkpoint2pb()
    pb2tflite()
