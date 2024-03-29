from train_ops import *
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

###########  Parameters  ############
#folder to save weights and images
DIR = 'experimentbeta'
BATCH_SIZE = 128
image_size = 192
epochs = 40
latent_dim = 25
lr = 1e-4
optimizer = tf.optimizers.Adam(lr)
# Extra optimizers for deep feature training
opt2 = tf.optimizers.Adam(lr)
opt3 = tf.optimizers.Adam(lr)
log_freq = 100
kernelsize = 3
# mode is one of: vae, dfc, combo, fixed, latent, beta
mode = 'combo'
# data is one of: mnist, celeba
data = 'celeba'
model = VAE(latent_dim, image_size, mode, kernelsize)
scales = {'kl_loss': 100, 'percep_loss': 1e6}
#####################################

if data == 'celeba':
    #input the celeb faces directory relative to the cwd
    image_dir='../img_align_celeba'
    all_image_paths=get_paths(image_dir)
    train_paths=all_image_paths[:-20000]
    test_paths=all_image_paths[-20000:]
    train_set= from_path_to_tensor(train_paths, BATCH_SIZE, size=image_size)
    test_set=from_path_to_tensor(test_paths, BATCH_SIZE, size=image_size)

if data == 'mnist':
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = tf.expand_dims(tf.constant(train_images, dtype=tf.float32), axis=3)
    test_images = tf.expand_dims(tf.constant(train_images, dtype=tf.float32), axis=3)
    TRAIN_BUF = 60000
    BATCH_SIZE = 100
    TEST_BUF = 10000
    train_set = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    test_set = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

train_dir='./{}/train'.format(DIR)
test_dir='./{}/test'.format(DIR)
# check if I'm about to overwrite event files
train_exists = os.path.exists(train_dir) and len(os.listdir(train_dir))!=0
test_exists = os.path.exists(test_dir) and len(os.listdir(test_dir))!=0
assert (not train_exists), "You are going to overwrite your train event files."
assert (not test_exists), "You are going to overwrite your test event files."
# Tensorboard logdirs
train_summary_writer = tf.summary.create_file_writer(train_dir)
test_summary_writer = tf.summary.create_file_writer(test_dir)

for epoch in range(1,epochs+1):
    if epoch != 1:
        train_set=train_set.shuffle(5000)
    start_time = time.time()
    for i, batch in enumerate(train_set):
        if epoch==1 and i<50:
            first_scales=scales
            first_scales['rc_loss']=1e4
            first_scales['kl_loss']=1
            loss_dict = train_step(batch, model, optimizer, opt2, opt3, mode, first_scales)
        else:
            loss_dict = train_step(batch, model, optimizer, opt2, opt3, mode, scales)
        if i==0:
            metrics_dict = {key: tf.metrics.Mean() for key in loss_dict}
        for loss, value in loss_dict.items():
            metrics_dict[loss].update_state(value)
        if tf.equal(optimizer.iterations % log_freq, 0):
            with train_summary_writer.as_default():
                for loss, metric in metrics_dict.items():
                    tf.summary.scalar(loss, metric.result(), step = optimizer.iterations)
                    metric.reset_states()

    with test_summary_writer.as_default():
        tester = test(loss_dict, image_size)
        avg_loss = tester(model, test_set, optimizer.iterations, mode, scales)
        print('Epoch: {}, test set average loss: {:.4f},'.format(epoch, avg_loss),
            'time elapsed for current epoch: {:.2f}'.format((time.time() - start_time)/60), 'minutes')
    if epoch % 10 == 0:
        model.saver(DIR, epoch)

rand_im = tf.random.normal(shape=(1,192,192,3))

_ = compute_loss(model, rand_im, mode, scales)

tf.saved_model.save(model, './{}/savedmodel'.format(DIR))
