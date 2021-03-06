# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:15:57 2018

@author: james
"""
import numpy as np
#'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
class DataGenerator:
    def __init__(self, location='dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',iterationNum=None):
        self.dataset_zip = np.load(location,encoding = 'latin1')
        print(self.dataset_zip.keys())
        self.imgs =  self.dataset_zip['imgs']
        self.latents_values =  self.dataset_zip['latents_values']
        self.latents_classes =  self.dataset_zip['latents_classes']
        self.metadata =  self.dataset_zip['metadata'][()]

        self.latents_possible_values = np.array([len(self.metadata['latents_possible_values'][x]) for x in self.metadata['latents_names']])
        self.n_samples = self.latents_possible_values[::-1].cumprod()[-1]
        self.image_shape = [self.imgs.shape[1],self.imgs.shape[2]]
        
        if iterationNum is None:
            iterationNum = int(self.n_samples/10)
        self.iterationNum= iterationNum
        
    def __iter__(self):
        self.i = 0
        while self.i+self.iterationNum <= self.imgs.shape[0]:
            yield self.imgs[self.i:self.i+self.iterationNum]
            self.i+=self.iterationNum
        if self.i == self.imgs.shape[0]:
            yield None
        if self.i+self.iterationNum > self.imgs.shape[0]:
            yield self.imgs[self.i+self.iterationNum:] 
            self.i= self.imgs.shape[0]
    
    def randomized(self):
        np.random.shuffle(self.imgs)
    
    def get_images(self,indices):
        images = []
        for index in indices:
            img = self.imgs[index]
            # img = img.reshape(4096)
            images.append(img)
        return np.array(images)

    def get_random_images(self, size):
        indices = [np.random.randint(self.n_samples) for i in range(size)]
        return self.get_images(indices)
    
    def generator_reset(self):
        self.i = 0
        
    def get_image(self, shape=0, scale=0, orientation=0, x=0, y=0):
        latents = [0, shape, scale, orientation, x, y]
        index = np.dot(latents, self.latents_possible_values).astype(int)
        return self.get_images([index])[0]

