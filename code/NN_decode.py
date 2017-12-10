from NN_main import get_model

def decode(config, w_vocab, t_vocab, batcher, t_op):

    batch_list = batcher.get_batch()

    decode_graph = tf.Graph()
    with tf.Session(graph=decode_graph) as sess:
        model = get_model(sess, config, decode_graph)

        for im_in, in_out in batcher.get_batch():
            out = model.dec_step(im_in)

    return out
