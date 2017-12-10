import time
import tensorflow as tf

from NN_model import ColorModel

def get_model(session, config, graph, mode='decode'):

    start_time = time.time()
    model = ColorModel(graph, config.l1_num_ofmaps, config.l2_num_ofmaps, config.lr)

    ckpt = tf.train.get_checkpoint_state(config.checkpoint_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        print("Time to restore model: %.2f" % (time.time() - start_time))
    elif mode == 'train':
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        print("Time to create model: %.2f" % (time.time() - start_time))
    else:
        raise ValueError('Model not found to restore.')
        return None
    return model
