# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 15:08:44 2018

@author: james
"""
import tensorflow as tf

class B_VAE:
    def __init__(self, gamma=100.0, capacity_limit=25.0, capacity_change_duration=100000, learning_rate=5e-4, stride = [1,2,2,1]):
        self.gamma = gamma
        self.capacity_limit = capacity_limit
        self.capacity_change_duration = capacity_change_duration
        self.learning_rate = learning_rate
        self.dtype = tf.float32
        self.conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=self.dtype)
        self.fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=self.dtype)
        self.stride = stride
        
    def _covnNet_weights(self,input_shape, name_space, deCovalutional = False):
        
        width = input_shape[0]
        height = input_shape[1]
        if deCovalutional:
            inChannels = input_shape[3]
            outChannels = input_shape[2]
        else:
            inChannels = input_shape[2]
            outChannels = input_shape[3]
        weight = tf.get_variable('conv_'+ str(name_space)+"_w", [width, height, inChannels, outChannels], initializer=self.conv_initializer, dtype=self.dtype)
        bias = tf.get_variable('conv_'+ str(name_space)+"_b", [outChannels], initializer=self.conv_initializer, dtype=self.dtype)
        return weight, bias
    
    def _full_connect_weights(self, input_shape, name_space):
        outChannels = input_shape[1]
        weight = tf.get_variable('full_'+ str(name_space)+"_w", input_shape, initializer=self.fc_initializer, dtype=self.dtype)
        bias = tf.get_variable('full_'+ str(name_space)+"_b", [outChannels], initializer=self.conv_initializer, dtype=self.dtype)
        return weight, bias
    
    def _denseNet(self, input, weight, bias, activation=True):
        if activation:
            return tf.nn.relu(tf.matmul(input,weight)+ bias)
        else:
            return tf.matmul(input,weight)+ bias
        
    def _covnNet(self,input,weight, bias, stride):
        conv_output = tf.nn.conv2d(input, weight, stride, 'SAME') + bias
        return tf.nn.relu(conv_output)
    
    def _encoder_network(self,input, dim , reuse=False):
        with tf.variable_scope("encode", reuse=reuse):
            input_shape = [3,3,1,dim[0]]
            weightcov, biascov = self._covnNet_weights(input_shape, 0)
            hidden = self._covnNet(input, weightcov,biascov)
            for x in range(1,len(dim)):
                input_shape = [3,3,dim[x-1],dim[x]]
                weightcov,biascov = self._covnNet_weights(input_shape, x)
                hidden = self._covnNet(hidden, weightcov ,biascov)
            hidden = tf.contrib.layers.flatten(hidden)
            weightfull, biasfull = self._full_connect_weights([hidden.get_shape()[0].value, 256], "fc1")
            fc1 = self._denseNet(hidden,weightfull, biasfull)
            mu_w, mu_b =  self._full_connect_weights([fc1.get_shape()[0].value, 10], "mu")
            mu = self._denseNet(fc1, mu_w, mu_b, activation=False)
            sigma_w, sigma_b =  self._full_connect_weights([fc1.get_shape()[0].value, 10], "sigma")
            sigma = self._denseNet(fc1,sigma_w, sigma_b, activation=False)
        return mu, sigma
    
    def _latent_sampling(self, z_mu, z_sigma):
        esplion = tf.random_normal(tf.shape(z_mu), 0,1, dtype= tf.float32)
        return tf.add(z_mu,tf.multiply(tf.sqrt(tf.exp(z_sigma)), esplion))
    
    def _generator_network(self, input, dim, reuse = False):
        with tf.variable_scope("decoder", reuse=reuse):
            weightfull, biasfull = self._full_connect_weights([self.mu.get_shape()[0].value, 256], "re_fc1")
            hidden = self._denseNet(input, weightfull, biasfull, activation=True)
            re_fc_shape =[hidden.get_shape()[1].value, 32].extend(self.image_shape)
            weightfull, biasfull = self._full_connect_weights(re_fc_shape, "re_fc2")
            hidden = self._denseNet(input, weightfull, biasfull, activation=True)
            hidden_reshape = tf.reshape(hidden, [-1, 3,3,re_fc_shape.get_shape()[1].value])
            
            input_shape = [3,3,32,dim[0]]
            weightcov, biascov = self._covnNet_weights(input_shape, "deconv_0", deCovalutional=True)
            hidden = self._deCovNet(hidden_reshape, weightcov,biascov,3,3,2)
            for x in range(1,len(dim)):
                input_shape = [3,3,dim[x-1],dim[x]]
                weightcov,biascov = self._covnNet_weights(input_shape,  "deconv_"+str(x), deCovalutional=True)
                hidden = self._covnNet(hidden, weightcov ,biascov)
            logit_shape = [-1].extend(self.image_shape)
            logit_out = tf.reshape(hidden, logit_shape )
        return logit_out
            
    def _deCovNet(self, input, weight , bias , window_width, window_height, stride , activation=True):
        out_channel   = weight.get_shape()[2].value
        out_height = window_height * stride
        out_width  = window_width * stride
        batch_size = tf.shape(input)[0]
        output_shape = tf.stack([batch_size, out_height, out_width, out_channel])
        deConv = tf.nn.conv2d_transpose(input, weight, output_shape, strides=[1, stride, stride, 1], padding='SAME') + bias
        if activation:
            return tf.nn.relu(deConv)
        return deConv
        
        
    def _network_init(self, input, encoder_dim , decoder_dim):
        self.x = tf.placeholder(tf.float32, input)
        self.image_shape = [input.get_shape()[1].value,input.get_shape()[2].value]
        with tf.varible_scope("Var_auto_encoder"):
            self.mu, self.sigma = self._encoder_network(self.x, encoder_dim)
            self.z = self._latent_sampling(self.mu, self.sigma)
            self.out = self._generator_network(self.z, decoder_dim)
            self.out_sigmoid = tf.nn.sigmoid(self.out)
        return
    
    def _loss_optimization(self):
        reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=self.out)
        reconstruction_loss = tf.reduce_sum(reconstruction_loss, 1)
        self.reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        
        

           
           