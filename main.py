# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:52:57 2018

@author: james
"""
import numpy as np
from DataGenerator import DataGenerator
from Beta_VAE import B_VAE
import tensorflow as tf
from tensorflow.python.platform import flags
from scipy.misc import imsave
import os

flag = flags.FLAGS

flags.DEFINE_float("gamma", 100.0, "Gemma for loss function")
flags.DEFINE_float("capacity_limit", 20.0, "Max capcaity number")
flags.DEFINE_float("capacity_duration", 10000, "Capcaity increaments by capacity_limit/ capacity_duration")
flags.DEFINE_boolean("training", True, "training or not")
flags.DEFINE_integer("epoch_num", 2000, "Epoch iteration")
flags.DEFINE_integer("batch_size", 64, "Batch size per iteration")
flags.DEFINE_string("log_file","./log_dir", "summary dir")
flags.DEFINE_string("checkpoint_dir","./check_point", "checkpoint dir")


def train(sess, model, dataGen,saver):
    summary_writer = tf.summary.FileWriter(flag.log_file, sess.graph)
    n_samples = dataGen.n_samples
    step = 0
    random_image_reconstruction = dataGen.get_random_images(10)
    batch_numbers = n_samples// dataGen.iterationNum
    for i in range(flag.epoch_num):
        reconstruction(sess,model, random_image_reconstruction)
        dataGen.randomized()
        dataGenerator = iter(dataGen)
        for x in range(batch_numbers):
            batch_Data = next(dataGenerator)
            reconstruction_loss, latent_loss, summary = model.batch_fit(batch_Data, sess,step)
            summary_writer.add_summary(summary, step)
            step += 1
        saver.save(sess, flag.checkpoint_dir + '/' + 'checkpoint', global_step = step)
        dataGen.generator_reset()
        
        disentable_test(sess, model, dataGen)

        
        
def reconstruction(sess, model, xs):
    x_out = model.generate_reconstruction(xs, sess)
    if not os.path.exists("reconstr_img"):
        os.mkdir("reconstr_img")
    for i in range(xs.shape[0]):
        org_img = xs[i].reshape(64,64)
        org_img = org_img.astype(np.float32)
        reconstr_img = x_out[i].reshape(64,64)
        imsave("./reconstr_img/org_{0}.png".format(i),      org_img)
        imsave("./reconstr_img/reconstr_{0}.png".format(i), reconstr_img)

def disentable_test(sess, model, dataGen):
    img = dataGen.get_image(shape=1, scale=2, orientation=5)
    batch_xs = [img]
    z_mean, z_sigma = model.transform(batch_xs,sess)
    z_sigma = np.exp(z_sigma)[0]
    z_mean = z_mean[0]
    zss_str = ""
    for i,zss in enumerate(z_sigma):
        str = "z{0}={1:.4f}, m{0}={1:.4f} ".format(i,zss,i, z_mean[i])
        zss_str += str + ", "
    print(zss_str)

def load_checkpoints(sess):
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(flag.checkpoint_dir)
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("loaded checkpoint: {0}".format(checkpoint.model_checkpoint_path))
  else:
    print("Could not find old checkpoint")
    if not os.path.exists(flag.checkpoint_dir):
      os.mkdir(flag.checkpoint_dir)
  return saver
        
def main():
    dataGen = DataGenerator(iterationNum=flag.batch_size)
    print(dataGen.iterationNum)
    random_image_reconstruction = dataGen.get_random_images(10)
    print(random_image_reconstruction.shape)
    model = B_VAE( [None, dataGen.image_shape[0], dataGen.image_shape[1]],gamma= flag.gamma, capacity_limit= flag.capacity_limit,capacity_change_duration= flag.capacity_duration)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = load_checkpoints(sess)
    if flag.training:
        train(sess,model,dataGen, saver)
        
if __name__ == '__main__':
    main()