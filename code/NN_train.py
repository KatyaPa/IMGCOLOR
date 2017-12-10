from NN_main import get_model

def train(config, cp_path):

    step_time, loss = 0.0, 0.0
    train_graph = tf.Graph()

    with tf.Session(graph=train_graph) as sess:
        model = get_model(sess, config, train_graph, 'train')

        current_step =  model.global_step.eval()
            for im_in, in_out in batcher.get_batch():
                start_time = time.time()
                # im_in, in_out = batcher.process(bv)
                step_loss, _  = model.step(sess, im_in, in_out)
                step_time += (time.time() - start_time)\
                                 / config.steps_per_checkpoint
                loss += step_loss / config.steps_per_checkpoint
                current_step += 1
                # Once in a while, we save checkpoint, print statistics
                if current_step % config.steps_per_checkpoint == 0:
                    # Print statistics for the previous epoch.
                    perplex = math.exp(loss) if loss < 300 else float('inf')
                    print ("global step %d learning rate %f step-time %.2f"
                           " perplexity %.6f (loss %.6f)" %
                           (model.global_step.eval(),
                           model.learning_rate.eval(),
                           step_time, perplex, loss))

                    # Save checkpoint and zero timer and loss.
                    ckpt_path = os.path.join(config.checkpoint_path, cp_path)
                    if not os.path.exists(config.checkpoint_path):
                        try:
                            os.makedirs(os.path.abspath(config.checkpoint_path))
                        except OSError as exc: # Guard against race condition
                            if exc.errno != errno.EEXIST:
                                raise
                    model.saver.save(sess, ckpt_path,
                                        global_step=model.global_step)
                    step_time, loss = 0.0, 0.0
                    sys.stdout.flush()
