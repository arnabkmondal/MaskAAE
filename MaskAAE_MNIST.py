import argparse
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
import math
from fid_score import evaluate_fid_score


SEED = 64
np.random.seed(SEED)
DPI = None
BIAS_INITIALIZER = None
KERNEL_INITIALIZER = tf.keras.initializers.he_uniform(seed=SEED)
FLAG = False
run_sess = tf.Session()
HEADER = '\033[95m'
OK_BLUE = '\033[94m'
OK_GREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
END_C = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return


def load_train_dataset(name, root_folder='./'):
    if name.lower() == 'mnist':
        (x, _), (_, _) = tf.keras.datasets.mnist.load_data()
        side_length = 28
        channels = 1
    elif name.lower() == 'fashion':
        (temp, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
        x = deepcopy(temp)
        side_length = 28
        channels = 1
    elif name.lower() == 'cifar10':
        (x, _), (_, _) = tf.keras.datasets.cifar10.load_data()
        side_length = 32
        channels = 3
    elif name.lower() == 'celeba140':
        data_folder = os.path.join(root_folder, 'data', name.lower())
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 64
        channels = 3
    elif name.lower() == 'celeba':
        data_folder = os.path.join(root_folder, 'data', name)
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 64
        channels = 3
    else:
        raise Exception('No such dataset called {}.'.format(name))
    np.random.shuffle(x)
    return x, side_length, channels


class TrainingDataGenerator:
    def __init__(self, x_train, side_length, channels):
        self.x_train = x_train / 255.0
        self.n_digits = self.x_train.shape[0]
        self.side_length = side_length
        self.channels = channels

    def get_batch(self, bs):
        image_indices = np.random.randint(0, self.n_digits, bs)
        return self.x_train[image_indices, :, :].reshape(-1, self.side_length, self.side_length, self.channels)


def image_grid(images, sv_path, dataset):
    size = int(images.shape[0] ** 0.5)
    images = (images - np.min(images)) / (np.max(images) - np.min(images))
    plt.figure(figsize=(20, 20))
    for i in range(images.shape[0]):
        plt.subplot(size, size, i + 1)
        image = images[i, :, :, :]
        if dataset == 'mnist' or dataset == 'fashion':
            image = np.reshape(image, [28, 28])
            plt.imshow(image, cmap='gray')
        elif dataset == 'celeba' or dataset == 'celeba140' or dataset == 'cifar10':
            plt.imshow(image)
        plt.axis('off')
    plt.savefig(sv_path, dpi=DPI)
    plt.close('all')
    return


def plot_graph(x, y, x_label, y_label, samples_dir, img_name, legend_list=[], n_col=1, active_dim=-1):
    plt.close('all')
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(labels=legend_list, loc='center left', bbox_to_anchor=(1, 0.5), ncol=n_col)
    plt.grid(b=True, which='both')
    plt.annotate('(Active Dimensions: {})'.format(active_dim), xy=(0.3, 0.5), xycoords='axes fraction')
    plt.savefig(samples_dir + img_name, dpi=DPI)


class MaskAAE:
    def __init__(self, dataset, bs, z_dim, lr, models_dir, samples_dir, training_steps, save_interval,
                 plot_interval, bn_axis, gradient_penalty_weight, variance_penalty_weight, disc_training_ratio, mask_on,
                 sess):
        self.dataset = dataset
        self.bs = bs
        self.z_dim = z_dim
        self.lr = lr
        self.model_dir = models_dir
        self.samples_dir = samples_dir
        self.training_steps = training_steps
        self.save_interval = save_interval
        self.plot_interval = plot_interval
        self.bn_axis = bn_axis
        self.gradient_penalty_weight = gradient_penalty_weight
        self.variance_penalty_weight = variance_penalty_weight
        self.disc_training_ratio = disc_training_ratio
        self.mask_on = mask_on
        self.is_training = tf.placeholder_with_default(False, (), 'is_training')
        self.sess = sess
        self.bias_initializer = None
        self.kernel_initializer = tf.compat.v1.glorot_normal_initializer()
        self.x_train, self.side_length, self.channels = load_train_dataset(name=dataset)
        self._build_model()
        self._loss()
        self._trainer()

    def _build_encoder(self, inp):
        encoded = inp

        if self.dataset == 'mnist':
            n_neurons = 1024
            encoded = tf.compat.v1.layers.flatten(inputs=encoded, name='enc-flatten')
            for i in range(4):
                encoded = tf.compat.v1.layers.dense(inputs=encoded, units=n_neurons, activation=None,
                                                    kernel_initializer=self.kernel_initializer,
                                                    bias_initializer=self.bias_initializer,
                                                    name='enc-dense-{}'.format(i))

                encoded = tf.nn.relu(encoded, name='enc-relu-{}'.format(i))
            encoded = tf.compat.v1.layers.dense(inputs=encoded, units=self.z_dim, activation=None,
                                                kernel_initializer=self.kernel_initializer,
                                                bias_initializer=self.bias_initializer, name='enc-final')
            return encoded

    def _build_decoder(self, inp):
        decoded = inp

        if self.dataset == 'mnist':
            n_neurons = 1024
            for i in range(4):
                decoded = tf.compat.v1.layers.dense(inputs=decoded, units=n_neurons, activation=None,
                                                    kernel_initializer=self.kernel_initializer,
                                                    bias_initializer=self.bias_initializer,
                                                    name='dec-dense-{}'.format(i))

                decoded = tf.nn.relu(decoded, name='dec-relu-{}'.format(i))
            decoded = tf.compat.v1.layers.dense(inputs=decoded, units=np.prod((28, 28, 1)), activation=tf.nn.sigmoid,
                                                kernel_initializer=self.kernel_initializer,
                                                bias_initializer=self.bias_initializer, name='dec-final')
            decoded = tf.reshape(tensor=decoded, shape=(-1, 28, 28, 1), name='dec-reshape')
            return decoded

    def _build_discriminator(self, inp):
        decision = inp
        n_neurons = 1024
        for i in range(4):
            decision = tf.compat.v1.layers.dense(inputs=decision, units=n_neurons, activation=tf.nn.relu,
                                                 kernel_initializer=self.kernel_initializer,
                                                 bias_initializer=self.bias_initializer,
                                                 name='disc-dense-{}'.format(str(i)))

        decision = tf.compat.v1.layers.dense(inputs=decision, units=1, activation=None,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer=self.bias_initializer, name='disc-final')

        return decision

    def _build_model(self):
        self.inp_img = tf.placeholder(tf.float32, [None, self.side_length, self.side_length, self.channels],
                                      name='input-image')
        self.prior_sample = tf.placeholder(tf.float32, [None, self.z_dim], name='prior-sample')
        if self.mask_on:
            self.mask_vec = tf.get_variable('Alpha', [self.z_dim], constraint=tf.keras.constraints.NonNeg())
            self.alpha = tf.ones_like(self.mask_vec, dtype=tf.float32, name='one-vector') - tf.exp(-self.mask_vec)
            self.active_dim = tf.math.count_nonzero(input_tensor=tf.greater_equal(self.alpha, 0.1),
                                                    dtype=tf.float32, name='active-z-dim')

            with tf.variable_scope('maae-encoder'):
                self.lat_vec = self._build_encoder(self.inp_img)
            self.masked_lat_vec = tf.multiply(self.lat_vec, self.alpha)
            with tf.variable_scope('maae-decoder', reuse=False):
                self.reconstructed_img = self._build_decoder(self.masked_lat_vec)
            self.masked_prior_sample = tf.multiply(self.prior_sample, self.alpha)
            with tf.variable_scope('maae-decoder', reuse=True):
                self.generated_img = self._build_decoder(self.masked_prior_sample)
            with tf.variable_scope('maae-discriminator', reuse=False):
                self.disc_op_for_lat_vec = self._build_discriminator(self.masked_lat_vec)
            with tf.variable_scope('maae-discriminator', reuse=True):
                self.disc_op_for_prior_sample = self._build_discriminator(self.masked_prior_sample)

            alpha = tf.random_uniform(
                shape=[self.bs, 1],
                minval=0.,
                maxval=1.,
                seed=SEED
            )
            differences = self.masked_lat_vec - self.masked_prior_sample
            interpolates = self.masked_prior_sample + (alpha * differences)
            with tf.variable_scope('maae-discriminator', reuse=True):
                gradients = tf.gradients(self._build_discriminator(interpolates), [interpolates])[0]
            self.slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))

        else:
            self.active_dim = tf.constant(self.z_dim)
            with tf.variable_scope('aae-encoder'):
                self.lat_vec = self._build_encoder(self.inp_img)
            with tf.variable_scope('aae-decoder', reuse=False):
                self.reconstructed_img = self._build_decoder(self.lat_vec)
            with tf.variable_scope('aae-decoder', reuse=True):
                self.generated_img = self._build_decoder(self.prior_sample)
            with tf.variable_scope('aae-discriminator', reuse=False):
                self.disc_op_for_lat_vec = self._build_discriminator(self.lat_vec)
            with tf.variable_scope('aae-discriminator', reuse=True):
                self.disc_op_for_prior_sample = self._build_discriminator(self.prior_sample)

            alpha = tf.random_uniform(
                shape=[self.bs, 1],
                minval=0.,
                maxval=1.,
                seed=SEED
            )
            differences = self.lat_vec - self.prior_sample
            interpolates = self.prior_sample + (alpha * differences)
            with tf.variable_scope('aae-discriminator', reuse=True):
                gradients = tf.gradients(self._build_discriminator(interpolates), [interpolates])[0]
            self.slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))

        return

    def _loss(self):
        self.reconstruction_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(self.inp_img,
                                                                                         self.reconstructed_img)
        # self.reconstruction_loss = tf.keras.losses.MeanSquaredError()(self.inp_img, self.reconstructed_img)
        self.generator_loss = -tf.reduce_mean(self.disc_op_for_lat_vec)
        self.discriminator_loss = tf.reduce_mean(self.disc_op_for_lat_vec) - tf.reduce_mean(
            self.disc_op_for_prior_sample)
        gradient_penalty = tf.reduce_mean((self.slopes - 1.) ** 2)
        self.discriminator_loss += self.gradient_penalty_weight * gradient_penalty
        if self.mask_on:
            mask_weight = tf.exp(-1 * 10 * tf.square(self.alpha))
            self.variance_penalty = tf.reduce_mean(tf.square(tf.reduce_mean(self.lat_vec, axis=0) - self.lat_vec)
                                                   * mask_weight)
            self.auto_encoder_loss = self.reconstruction_loss + self.variance_penalty_weight * self.variance_penalty
            poly_reg = tf.reduce_sum(tf.math.abs(tf.multiply((tf.ones([self.z_dim], tf.float32) - self.alpha),
                                                             self.alpha)))
            self.poly_reg_coeff = tf.placeholder(tf.float32, [1], name='poly-reg-coeff')
            self.mask_loss = 1000 * self.reconstruction_loss + \
                             tf.square(1 - (tf.reduce_mean(self.disc_op_for_lat_vec) -
                                            tf.reduce_mean(self.disc_op_for_prior_sample))) + \
                             self.poly_reg_coeff * poly_reg
        else:
            self.auto_encoder_loss = self.reconstruction_loss

        return

    def _trainer(self):
        self.ae_lr_decay = tf.placeholder(tf.float32, [1], name='ae-lr-decay')
        self.gen_lr_decay = tf.placeholder(tf.float32, [1], name='gen-lr-decay')
        self.disc_lr_decay = tf.placeholder(tf.float32, [1], name='disc-lr-decay')
        self.mask_lr_decay = tf.placeholder(tf.float32, [1], name='mask-lr-decay')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if self.mask_on:
            self.auto_encoder_variables = [var for var in tf.compat.v1.trainable_variables() if
                                           (var.name.startswith('maae-encoder') or
                                            var.name.startswith('maae-decoder'))]
            self.generator_variables = [var for var in tf.compat.v1.trainable_variables() if
                                        var.name.startswith('maae-encoder')]
            self.discriminator_variables = [var for var in tf.compat.v1.trainable_variables() if
                                            var.name.startswith('maae-discriminator')]
            self.mask_variables = [var for var in tf.compat.v1.trainable_variables() if
                                   var.name.startswith('Alpha')]

            with tf.control_dependencies(update_ops):
                self.mask_trainer = tf.train.RMSPropOptimizer(5 * self.lr) \
                    .minimize(self.mask_loss * self.mask_lr_decay, var_list=self.mask_variables)

        else:
            self.auto_encoder_variables = [var for var in tf.compat.v1.trainable_variables() if
                                           (var.name.startswith('aae-encoder') or
                                            var.name.startswith('aae-decoder'))]
            self.generator_variables = [var for var in tf.compat.v1.trainable_variables() if
                                        var.name.startswith('aae-encoder')]
            self.discriminator_variables = [var for var in tf.compat.v1.trainable_variables() if
                                            var.name.startswith('aae-discriminator')]

        with tf.control_dependencies(update_ops):
            self.auto_encoder_trainer = tf.train.RMSPropOptimizer(5 * self.lr) \
                .minimize(self.auto_encoder_loss * self.ae_lr_decay, var_list=self.auto_encoder_variables)
            self.generator_trainer = tf.train.RMSPropOptimizer(self.lr).minimize(
                self.generator_loss * self.gen_lr_decay, var_list=self.generator_variables)
            self.discriminator_trainer = tf.train.RMSPropOptimizer(self.lr) \
                .minimize(self.discriminator_loss * self.disc_lr_decay, var_list=self.discriminator_variables)

        return

    def train(self):
        sess = self.sess
        saver = tf.train.Saver(max_to_keep=(self.training_steps // self.save_interval))
        sess.run(tf.global_variables_initializer())
        if self.mask_on:
            sess.run(self.mask_vec.assign(
                tf.compat.v1.random.uniform(shape=[self.z_dim], minval=0.0, maxval=5.0, seed=SEED,
                                            name='mask-initializer')))

            ae_training_ratio = 1
            reg_schedule_interval = 75 if self.z_dim <= 20 else min(10000, math.floor(1125 * self.z_dim / 10))

            mask_buf = []
            mask_loss_buf = []
            var_buf = []
            variance_penalty_buf = []
        else:
            ae_training_ratio = 1
            reg_schedule_interval = 0

        ae_loss_buf = []
        reconstruction_buf = []
        disc_loss_buf = []
        generator_loss_buf = []
        steps_buf = []
        legend_list = ['z' + str(i) for i in range(self.z_dim)]
        auto_encoder_lr_decay_val = np.asarray([1])
        generator_lr_decay_val = np.asarray([1])
        discriminator_lr_decay_val = np.asarray([1])
        mask_lr_decay_val = np.asarray([1])
        poly_reg_wt = np.asarray([2 / self.z_dim])
        true_data_gen = TrainingDataGenerator(self.x_train, self.side_length, self.channels)

        for step in range(self.training_steps):
            for _ in range(ae_training_ratio):
                image_batch = true_data_gen.get_batch(bs=self.bs)
                gaussian_sample = np.random.normal(0, 1, (self.bs, self.z_dim))
                ae_feed_dict = {self.inp_img: image_batch, self.prior_sample: gaussian_sample,
                                self.ae_lr_decay: auto_encoder_lr_decay_val, self.gen_lr_decay: generator_lr_decay_val,
                                self.disc_lr_decay: discriminator_lr_decay_val, self.mask_lr_decay: mask_lr_decay_val,
                                self.is_training: True}
                sess.run(self.auto_encoder_trainer, feed_dict=ae_feed_dict)

            for _ in range(self.disc_training_ratio):
                image_batch = true_data_gen.get_batch(bs=self.bs)
                gaussian_sample = np.random.normal(0, 1, (self.bs, self.z_dim))
                disc_feed_dict = {self.inp_img: image_batch, self.prior_sample: gaussian_sample,
                                  self.ae_lr_decay: auto_encoder_lr_decay_val,
                                  self.gen_lr_decay: generator_lr_decay_val,
                                  self.disc_lr_decay: discriminator_lr_decay_val, self.mask_lr_decay: mask_lr_decay_val,
                                  self.is_training: True}
                sess.run(self.discriminator_trainer, feed_dict=disc_feed_dict)

            image_batch = true_data_gen.get_batch(bs=self.bs)
            gen_feed_dict = {self.inp_img: image_batch, self.prior_sample: gaussian_sample,
                             self.ae_lr_decay: auto_encoder_lr_decay_val,
                             self.gen_lr_decay: generator_lr_decay_val,
                             self.disc_lr_decay: discriminator_lr_decay_val, self.mask_lr_decay: mask_lr_decay_val,
                             self.is_training: True}
            sess.run(self.generator_trainer, feed_dict=gen_feed_dict)

            if self.mask_on:
                image_batch = true_data_gen.get_batch(bs=self.bs)
                gaussian_sample = np.random.normal(0, 1, (self.bs, self.z_dim))
                if step % (4 * reg_schedule_interval) == 0 and step > 0:
                    poly_reg_wt = np.asarray([min(2048, poly_reg_wt[0] * 2)])
                mask_feed_dict = {self.inp_img: image_batch, self.prior_sample: gaussian_sample,
                                  self.ae_lr_decay: auto_encoder_lr_decay_val,
                                  self.gen_lr_decay: generator_lr_decay_val,
                                  self.disc_lr_decay: discriminator_lr_decay_val,
                                  self.mask_lr_decay: mask_lr_decay_val,
                                  self.poly_reg_coeff: poly_reg_wt,
                                  self.is_training: True}
                sess.run(self.mask_trainer, feed_dict=mask_feed_dict)

            if step == 5 * reg_schedule_interval:
                auto_encoder_lr_decay_val = auto_encoder_lr_decay_val / 5
                mask_lr_decay_val = mask_lr_decay_val / 5

            if step % (self.plot_interval // 10) == 0:
                image_batch = true_data_gen.get_batch(bs=self.bs)
                gaussian_sample = np.random.normal(0, 1, (self.bs, self.z_dim))
                feed_dict = {self.inp_img: image_batch, self.prior_sample: gaussian_sample,
                             self.ae_lr_decay: auto_encoder_lr_decay_val, self.gen_lr_decay: generator_lr_decay_val,
                             self.disc_lr_decay: discriminator_lr_decay_val, self.is_training: False}
                if self.mask_on:
                    feed_dict.update({self.mask_lr_decay: mask_lr_decay_val, self.poly_reg_coeff: poly_reg_wt})
                    reconstructed_images = sess.run(self.reconstructed_img, feed_dict=feed_dict)
                    image_grid(images=image_batch, sv_path='{}true_iter_{}.png'.format(self.samples_dir, step),
                               dataset=self.dataset)
                    image_grid(images=reconstructed_images, sv_path='{}recon_iter_{}.png'.format(self.samples_dir, step),
                               dataset=self.dataset)
                    generated_images = sess.run(self.generated_img, feed_dict=feed_dict)
                    image_grid(images=generated_images, sv_path='{}generated_iter_{}.png'.format(self.samples_dir, step),
                               dataset=self.dataset)
                ae_loss_val = sess.run(self.auto_encoder_loss, feed_dict=feed_dict)
                disc_loss_val = sess.run(self.discriminator_loss, feed_dict=feed_dict)
                generator_loss_val = sess.run(self.generator_loss, feed_dict=feed_dict)
                reconstruction_buf.append(sess.run(self.reconstruction_loss, feed_dict=feed_dict))
                ae_loss_buf.append(ae_loss_val)
                disc_loss_buf.append(disc_loss_val)
                generator_loss_buf.append(generator_loss_val)
                steps_buf.append(step)

                if self.mask_on:
                    mask_buf.append(sess.run(self.alpha))
                    mask_loss_val = sess.run(self.mask_loss, feed_dict=feed_dict)
                    mask_loss_buf.append(mask_loss_val[0])
                    variance_penalty_buf.append(sess.run(self.variance_penalty, feed_dict=feed_dict))
                    lat_vec_arr = sess.run(self.lat_vec, feed_dict=feed_dict)
                    lat_vec_var = np.var(lat_vec_arr, axis=0)
                    var_buf.append(lat_vec_var)
                    active_z_dim = int(sess.run(self.active_dim))
                    print(HEADER + 'Training an MAAE Model on {}'.format(self.dataset.upper()) + END_C)
                    print(WARNING + 'Step: {}/{}, '.format(steps_buf[-1], self.training_steps) + END_C + OK_BLUE +
                          'Reconstruction Loss: {}, Variance Loss: {}, AE Loss: {}, Disc Loss: {}, Gen Loss: {}, '
                          'Mask Loss: {}, Number of Active Dimensions: {}/{}'.
                          format(reconstruction_buf[-1], variance_penalty_buf[-1], ae_loss_buf[-1], disc_loss_buf[-1],
                                 generator_loss_buf[-1], mask_loss_buf[-1], active_z_dim, self.z_dim)
                          + END_C + '\n')
                else:
                    active_z_dim = self.z_dim
                    print(HEADER + 'Training an AAE Model on {}'.format(self.dataset.upper()) + END_C)
                    print(WARNING + 'Step: {}/{}, '.format(steps_buf[-1], self.training_steps) + END_C + OK_BLUE +
                          'Reconstruction Loss: {}, AE Loss: {}, Disc Loss: {}, Gen Loss: {}, Latent Dim: {}'.
                          format(reconstruction_buf[-1], ae_loss_buf[-1], disc_loss_buf[-1], generator_loss_buf[-1],
                                 self.z_dim)
                          + END_C + '\n')

            if step % self.plot_interval == 0 and step > 0:
                plot_graph(x=steps_buf, y=reconstruction_buf, x_label='Steps', y_label='Reconstruction Loss',
                           samples_dir=self.samples_dir, img_name='reconstruction_loss.png', legend_list=[],
                           active_dim=active_z_dim)
                plot_graph(x=steps_buf, y=ae_loss_buf, x_label='Steps', y_label='AE Loss', samples_dir=self.samples_dir,
                           img_name='ae_loss.png', legend_list=[], active_dim=active_z_dim)
                plot_graph(x=steps_buf, y=disc_loss_buf, x_label='Steps', y_label='Discriminator Loss',
                           samples_dir=self.samples_dir, img_name='disc_loss.png', legend_list=[],
                           active_dim=active_z_dim)
                plot_graph(x=steps_buf, y=generator_loss_buf, x_label='Steps', y_label='Generator Loss',
                           samples_dir=self.samples_dir, img_name='gen_loss.png', legend_list=[],
                           active_dim=active_z_dim)
                if self.mask_on:
                    plot_graph(x=steps_buf, y=mask_buf, x_label='Steps', y_label='Mask Value',
                               samples_dir=self.samples_dir, img_name='mask_trace.png',
                               legend_list=legend_list, n_col=4, active_dim=active_z_dim)
                    plot_graph(x=steps_buf, y=mask_loss_buf, x_label='Steps', y_label='Mask Loss Value',
                               samples_dir=self.samples_dir, img_name='mask_loss.png', active_dim=active_z_dim)
                    plot_graph(x=steps_buf, y=var_buf, x_label='Steps', y_label='Variance Value',
                               samples_dir=self.samples_dir, img_name='var_trace.png', legend_list=legend_list,
                               n_col=4, active_dim=active_z_dim)
                    plot_graph(x=steps_buf, y=variance_penalty_buf, x_label='Steps', y_label='Variance Loss',
                               samples_dir=self.samples_dir, img_name='variance_loss.png',
                               legend_list=['Variance Penalty'], active_dim=active_z_dim)

            if step % self.save_interval == 0 and step > 0:
                if self.mask_on:
                    saver.save(sess, os.path.join(self.model_dir, 'maae.model'), global_step=step)
                else:
                    saver.save(sess, os.path.join(self.model_dir, 'aae.model'), global_step=step)

        with open(self.samples_dir + 'reconstruction_loss.txt', 'wb') as fp:
            pickle.dump(reconstruction_buf, fp)
        with open(self.samples_dir + 'ae_loss.txt', 'wb') as fp:
            pickle.dump(ae_loss_buf, fp)
        with open(self.samples_dir + 'disc_loss.txt', 'wb') as fp:
            pickle.dump(disc_loss_buf, fp)
        with open(self.samples_dir + 'generator_loss.txt', 'wb') as fp:
            pickle.dump(generator_loss_buf, fp)
        with open(self.samples_dir + 'plot_steps.txt', 'wb') as fp:
            pickle.dump(steps_buf, fp)

        if self.mask_on:
            with open(self.samples_dir + 'mask_value.txt', 'wb') as fp:
                pickle.dump(mask_buf, fp)
            with open(self.samples_dir + 'mask_loss_value.txt', 'wb') as fp:
                pickle.dump(mask_loss_buf, fp)
            with open(self.samples_dir + 'variance_value.txt', 'wb') as fp:
                pickle.dump(var_buf, fp)
            with open(self.samples_dir + 'variance_penalty.txt', 'wb') as fp:
                pickle.dump(variance_penalty_buf, fp)

        return


def generate_images(model_base_path, step, dataset, batch_size, z_dim, gen_img_dir, mask_on):
    tf.reset_default_graph()
    sess = tf.Session()
    if mask_on:
        meta_file = model_base_path + 'maae.model-' + str(step) + '.meta'
        saver = tf.train.import_meta_graph(meta_file)
        with open(model_base_path + 'checkpoint') as fp:
            content = fp.readlines()
            content[0] = 'model_checkpoint_path: "maae.model-{}"\n'.format(str(step))
        with open(model_base_path + 'checkpoint', 'w') as fp:
            fp.writelines(content)
        saver.restore(sess, tf.train.latest_checkpoint(model_base_path))
        graph = tf.get_default_graph()
        latent_vec = graph.get_tensor_by_name('maae-encoder/enc-final/BiasAdd:0')
        generated_image = graph.get_tensor_by_name("maae-decoder/dec-final/Sigmoid:0")

    else:
        meta_file = model_base_path + 'aae.model-' + str(step) + '.meta'
        saver = tf.train.import_meta_graph(meta_file)
        with open(model_base_path + 'checkpoint') as fp:
            content = fp.readlines()
            content[0] = 'model_checkpoint_path: "aae.model-{}"\n'.format(str(step))
        with open(model_base_path + 'checkpoint', 'w') as fp:
            fp.writelines(content)
        saver.restore(sess, tf.train.latest_checkpoint(model_base_path))
        graph = tf.get_default_graph()
        latent_vec = graph.get_tensor_by_name('aae-encoder/enc-final/BiasAdd:0')
        generated_image = graph.get_tensor_by_name("aae-decoder/dec-final/Sigmoid:0")

    z_sample = np.random.normal(0, 1, (batch_size, z_dim))
    if dataset == 'mnist' or dataset == 'fashion':
        gen_img = sess.run(generated_image, feed_dict={latent_vec: z_sample})
        gen_img = gen_img.reshape(-1, 28, 28, 1)
    else:
        gen_img = sess.run(generated_image, feed_dict={latent_vec: z_sample})
    image_grid(images=gen_img, sv_path='{}sample_generated_iter_{}.png'.format(gen_img_dir, step), dataset=dataset)

    z_sample = np.random.normal(0, 1, (10000, z_dim))
    feed_dict = {latent_vec: z_sample}
    if dataset == 'mnist' or dataset == 'fashion':
        gen_img = sess.run(generated_image, feed_dict=feed_dict)
        gen_img = gen_img.reshape(-1, 28, 28, 1)
    elif dataset == 'cifar10':
        gen_img = sess.run(generated_image, feed_dict=feed_dict)
    elif dataset == 'celeba' or dataset == 'celeba140':
        gen_img = np.ndarray(shape=(10000, 64, 64, 3))
        for j in range(10):
            z_sample = np.random.normal(0, 1, (1000, z_dim))
            feed_dict = {latent_vec: z_sample}
            gen_img_batch = sess.run(generated_image, feed_dict=feed_dict)
            gen_img[(j * 1000):((j + 1) * 1000)] = gen_img_batch

    np.save('{}generated_{}_images_{}.npy'.format(gen_img_dir, dataset, str(step)), gen_img)

    return


def evaluate_fid(generated_image_dir, dataset, step, root_folder='./'):
    tf.reset_default_graph()
    fake_images = np.load('{}generated_{}_images_{}.npy'.format(generated_image_dir, dataset, step))
    try:
        fid_val = evaluate_fid_score(fake_images, dataset, root_folder, True)
    except:
        fid_val = np.nan
    return fid_val


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist'])
parser.add_argument('--z_dim', type=int, default=48)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save_interval', type=int, default=5000)
parser.add_argument('--plot_interval', type=int, default=1000)
parser.add_argument('--training_steps', type=int, default=500000)
parser.add_argument('--bn_axis', type=int, default=-1)
parser.add_argument('--mask_on', type=int, default=1, choices=[0, 1])
parser.add_argument('--gradient_penalty_weight', type=float, default=10)
parser.add_argument('--variance_penalty_weight', type=float, default=100)
parser.add_argument('--disc_training_ratio', type=int, default=5)

args = parser.parse_args()
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

if args.mask_on:
    trained_models_dir = f'./maae_models/{args.dataset}/zdim{args.z_dim}/'
    training_data_dir = f'./maae_samples/{args.dataset}/zdim{args.z_dim}/'
    generated_data_dir = f'./maae_generated/{args.dataset}/zdim{args.z_dim}/'
    create_directory(trained_models_dir)
    create_directory(training_data_dir)
    create_directory(generated_data_dir)
else:
    trained_models_dir = f'./aae_models/{args.dataset}/zdim{args.z_dim}/'
    training_data_dir = f'./aae_samples/{args.dataset}/zdim{args.z_dim}/'
    generated_data_dir = f'./aae_generated/{args.dataset}/zdim{args.z_dim}/'
    create_directory(trained_models_dir)
    create_directory(training_data_dir)
    create_directory(generated_data_dir)

tf.reset_default_graph()
model = MaskAAE(dataset=args.dataset, bs=args.batch_size, z_dim=args.z_dim, lr=args.lr,
                models_dir=trained_models_dir, samples_dir=training_data_dir, training_steps=args.training_steps,
                save_interval=args.save_interval, plot_interval=args.plot_interval, bn_axis=args.bn_axis,
                gradient_penalty_weight=args.gradient_penalty_weight,
                variance_penalty_weight=args.variance_penalty_weight, disc_training_ratio=args.disc_training_ratio,
                mask_on=args.mask_on, sess=tf.Session())
model.train()

print(HEADER + 'Training Complete' + END_C)
print(HEADER + 'Generating Images' + END_C)

for i in range(args.save_interval, args.training_steps, args.save_interval):
    generate_images(model_base_path=trained_models_dir, step=i, dataset=args.dataset, batch_size=args.batch_size,
                    z_dim=args.z_dim, gen_img_dir=generated_data_dir, mask_on=args.mask_on)

print(HEADER + 'Evaluation FID Score' + END_C)

fid_buf = []
step_buf = []
for i in range(args.save_interval, args.training_steps, args.save_interval):
    tf.reset_default_graph()
    fid_val = evaluate_fid(generated_image_dir=generated_data_dir, dataset=args.dataset, step=i, root_folder='./')
    print(i, fid_val)
    fid_buf.append(fid_val)
    step_buf.append(i)
with open(generated_data_dir + 'fid_buf.txt', 'wb') as f:
    pickle.dump(fid_buf, f)
with open(generated_data_dir + 'iteration_buf.txt', 'wb') as f:
    pickle.dump(step_buf, f)
print(fid_buf)
min_fid = min(fid_buf)
min_fid_step = step_buf[fid_buf.index(min_fid)]
plt.close('all')
plt.plot(step_buf, fid_buf)
plt.scatter(min_fid_step, min_fid, color='r')
plt.annotate('({}, {})'.format(min_fid_step, min_fid), xy=(0.6, 0.95), xycoords='axes fraction')
plt.xlabel('Iterations')
plt.ylabel('FID')
plt.savefig(f'{generated_data_dir}fid_plot_{SEED}.png', dpi=None)
