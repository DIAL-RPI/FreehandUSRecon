#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:24:11 2018

@author: haskig
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 12:55:09 2018

@author: labadmin
"""

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, ZeroPadding3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, UpSampling3D, Conv2D, Conv3D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import SimpleITK as sitk
from os import path

import matplotlib.pyplot as plt

import sys

import numpy as np
from stl import mesh

"""
from utils import volume_resampler_3d_no_trans as vr3D
from utils import reg_evaluator as regev
"""


class distributions():
    def __init__(self, n):
        self.n=n
    
    def get_path_US(self, m):
        #n refers to the case number. This will output a path
        folder = '/home/haskig/data/uronav_data'
        return path.join(folder, 'Case{:04}/USVol.mhd'.format(m))
    
    def get_path_MR(self, m):
        #n refers to the case number. This will output a path
        folder = '/home/haskig/data/uronav_data'
        return path.join(folder, 'Case{:04}/MRVol.mhd'.format(m))
    
    def get_segmentation(self, m):
        folder = '/home/haskig/data/uronav_data'
        path_seg = path.join(folder, 'Case{:04}/segmentationrtss.uronav.stl'.format(m))
        return mesh.Mesh.from_file(path_seg)
    
    def load_itk(self, enter_path):
        itkimage=sitk.ReadImage(enter_path)
        image = sitk.Cast(itkimage, sitk.sitkUInt8)
        #img = sitk.GetArrayFromImage(image)
        
        return image
    
    def get_reg(self, m):
        folder = '/home/data/uronav_data/Case{:04}'.format(m)
        fn_reg = path.join(folder, 'coreg.txt')
        evaluator = regev.RegistrationEvaluator(folder)
        return evaluator.load_registration(fn_reg)
    
    def get_distributions(self):
        #get every volume for cases 1 to n     
        US_vols = []
        MR_vols = []
        DNE_count = 0
        for i in range(1,self.n+1):
            try:
                path_US = self.get_path_US(i)
                path_MR = self.get_path_MR(i)
                seg = self.get_segmentation(i)
                m2f_transform = self.get_reg(i)
                US_vol = self.load_itk(path_US)
                MR_vol = self.load_itk(path_MR)
                vr3d = vr3D.VolumeResampler(MR_vol, seg, 
                                                US_vol, m2f_transform,
                                                enlargeRatio = 0.3)
                MR_vol, US_vol = vr3d.resample(96,96,32)
                MR_vol = sitk.GetArrayFromImage(MR_vol)
                US_vol = sitk.GetArrayFromImage(US_vol)
                US_vol.tolist()
                MR_vol.tolist()
                US_vols.append(US_vol)
                MR_vols.append(MR_vol)
            except (RuntimeError, FileNotFoundError):
                DNE_count += 1
        US_vols = np.asarray(US_vols)
        MR_vols = np.asarray(MR_vols)
        #print(str(DNE_count) + ' cases out of the ' + str(self.n) + ' have been deleted!')
    
        return US_vols, MR_vols
    
Dist = distributions(749)
US_train, MR_train = Dist.get_distributions()

US_path = Dist.get_path_US(1)
US = Dist.load_itk(US_path)
US = sitk.GetArrayFromImage(US)

idx = 30
plt.imshow(US[idx], cmap='gray')
plt.show()


class DCGAN():
    def __init__(self):
        
        #self.X_train=X_train
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_cols, self.img_rows, self.channels)
        self.latent_dim = 100
        
        self.num_cases = 749
        

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
  

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        #model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        #model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        #X_train = self.X_train

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)


    def save_imgs(self, epoch):
        #figure out how to adapt this function to our dataset
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("/home/haskig/Pictures/mnist_epoch{}".format(epoch))
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
dcgan.train(epochs=4000, batch_size=32, save_interval=50)

for i in range(1,750):
    try:
       if sitk.GetArrayFromImage(Dist.load_itk(Dist.get_path_MR(i))).shape == (28, 28, 28, 1):
            print('Shape found for case {:04}'.format(i))
    except (RuntimeError, FileNotFoundError):
        pass
