from train import *

if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        #folder to save weights and images
        DIR='dist_train1'

        #input the celeb faces directory relative to the cwd
        image_dir='../img_align_celeba'

        all_image_paths=get_paths(image_dir)
        image_count=len(all_image_paths)

        train_paths=all_image_paths[:-20000]
        test_paths=all_image_paths[-20000:]

        BATCH_SIZE = 128

        train_set= from_path_to_tensor(train_paths, BATCH_SIZE)
        test_set=from_path_to_tensor(test_paths, BATCH_SIZE)

        # check if I'm about to overwrite event files
        train_dir='./{}/summaries/train'.format(DIR)
        test_dir='./{}/summaries/train'.format(DIR)
        train_exists = os.path.exists(train_dir) and len(os.listdir(train_dir))!=0
        test_exists = os.path.exists(test_dir) and len(os.listdir(test_dir))!=0
        assert (not train_exists), "You are going to overwrite your train event files."
        assert (not test_exists), "You are going to overwrite your test event files."

        train_summary_writer = tf.summary.create_file_writer(TRAINING_DIR+'/summaries/train')
        test_summary_writer = tf.summary.create_file_writer(TRAINING_DIR+'/summaries/test')

        epochs = 10
        latent_dim = 50
        num_examples_to_generate = 4

        optimizer=tf.optimizers.Adadelta(1e-4)

        # to be used for checking progress.
        random_vector_for_generation = tf.random.normal(
            shape=[num_examples_to_generate, latent_dim])

        log_freq=10

        model = VAE(latent_dim)

        for epoch in range(1,epochs+1):
            start_time = time.time()
            rcmetric = tf.metrics.Mean()
            klmetric = tf.metrics.Mean()
            totalmetric  = tf.metrics.Mean()
            for batch in train_set:
                rc_loss, kl_loss, loss = train_step(batch, model, optimizer)
                rcmetric.update_state(rc_loss)
                klmetric.update_state(kl_loss)
                totalmetric.update_state(total_loss)
                if tf.equal(optimizer.iterations % log_freq, 0):
                    with train_summary_writer:
                        tf.summary.scalar('rc_loss', rcmetric.result(), step = step)
                        tf.summary.scalar('kl_loss', klmetric.result(), step = step)
                        tf.summary.scalar('total_loss', totalmetric.result(), step = step)
                    rcmetric.reset_states()
                    klmetric.reset_states()
                    totalmetric.reset_states()
            with test_summary_writer:
                avg_loss = test(model, test_set, optimizer.iterations)
                print('Epoch: {}, test set average loss: {},'.format(epoch, avg_loss),
                    'time elapse for current epoch {}'.format(time.time() - start_time))
            tf.saved_model.save(model, './{}/{}'.format(DIR,epoch))
