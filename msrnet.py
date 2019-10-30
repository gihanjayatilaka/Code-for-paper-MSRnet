import tensorflow as tf
import keras
import keras.backend as K
import numpy as np
from keras.layers import Conv2D,Activation,Input,Lambda,Concatenate
from keras.models import Model
import os
import cv2
import sys
from ImageHandler import ImageHandler
from keras.models import load_model
from keras import regularizers
import numpy as np


L2_REG=False
REG_COEFF=0.01
EPS=0.0001

def logFunc(x):
   return K.log(x)

def nn(x,outputLayerAct):


    print('X',x.shape)

    v=[1.0, 10.0, 100.0, 300.0]
    # v=[1.0,1.0,1.0,1.0]
    x1=[]

    for i in range(len(v)):
        x1.append(Lambda(lambda a: 1.0+ x*v[i])(x))
        # x1[i]=Activation('relu')(x1[i])
        # x1[i] = Lambda(lambda a: a)(x1[i])
        x1[i]=(Lambda((lambda a: logFunc(a)))(x1[i]))
        # xx1.append(K.log(x1[i]))

    print(x1)
    x1=Concatenate(axis=-1)(x1)
    print('x1',x1.shape)
    if L2_REG:
        x1=Conv2D(32,(1,1),padding='same',kernel_regularizer=regularizers.l2(REG_COEFF))(x1)
    else:
        x1 = Conv2D(32, (1, 1), padding='same')(x1)
    x1=Activation('relu')(x1)

    if L2_REG:
        x1=Conv2D(3,(3,3),padding='same',name='X1',kernel_regularizer=regularizers.l2(REG_COEFF))(x1)
    else:
        x1 = Conv2D(3, (3, 3), padding='same', name='X1')(x1)

    print("x1",x1.shape)


    x2=[]
    if L2_REG:
        x2.append(Conv2D(3,(1,1),padding='same',kernel_regularizer=regularizers.l2(REG_COEFF))(x1))
    else:
        x2.append(Conv2D(3, (1, 1), padding='same')(x1))
    x2[0]=Activation('relu')(x2[0])
    for l in range(1,10):
        if L2_REG:
            x2.append(Conv2D(32, (3, 3),padding='same',kernel_regularizer=regularizers.l2(REG_COEFF))(x2[l-1]))
        else:
            x2.append(Conv2D(32, (3, 3), padding='same')(x2[l - 1]))
        x2[l] = Activation('relu')(x2[l])

    x2=Concatenate(axis=-1)(x2)
    print("x2",x2.shape)


    if L2_REG:
        x2=Conv2D(3,(1,1),padding='same',name='X2',kernel_regularizer=regularizers.l2(REG_COEFF))(x2)
    else:
        x2 = Conv2D(3, (1, 1), padding='same', name='X2')(x2)
    # x2=Activation('relu')(x2)
    print("x2", x2.shape)

    x2 = Lambda(lambda a: a[0] - a[1],name='DIFF')([x1, x2])

    print("(diff) x2", x2.shape)
    x3=None
    if L2_REG:
        x3=Conv2D(3,(1,1),padding='same',kernel_regularizer=regularizers.l2(REG_COEFF))(x2)
    else:
        x3 = Conv2D(3, (1, 1), padding='same')(x2)

    if outputLayerAct=='relu':
        x3=Activation('relu')(x3)
        input("Relu act?")
    elif outputLayerAct=='sigmoid':
        x3=Activation('sigmoid')(x3)
        input("Sig act?")
    else:
        input("No act?")
    # x3=Conv2D(3,(1,1),padding='same')(x3)

    print("x3",x3.shape)

    # loss=tf.reduce_mean(tf.square(x3-x))
    # print("Loss",loss.shape)

    return x3


def splitInto4(imgAr):
    # imgAr[N,H,W,3]
    newImg=np.ndarray((1000,64*6,64*4,3),dtype=np.uint8)
    newImg.fill(0)
    for i in range(imgAr.shape[0]):
        newImg[i,:,:64,:]=imgAr[i,:,:,:]
        newImg[i, :, 64:2*64, 0] = imgAr[i, :, :, 0]
        newImg[i, :, 2*64:3 * 64, 1] = imgAr[i, :, :, 1]
        newImg[i, :, 3*64:4 * 64, 2] = imgAr[i, :, :, 2]

    return newImg



if __name__ == '__main__':
    print("python msrnet.py mod.h5 None|relu|sigmoid aa.npz")
    MODEL_FILE_NAME=sys.argv[1]
    actFunc=sys.argv[2]
    npzSaveFile=sys.argv[3]



    img_hndlr = ImageHandler((64, 64))

    path = "dataset"

    if not (os.path.exists(path + "/dark/") and os.path.exists(path + "/true/")):
        img_hndlr.create_dataset(path)

    X = img_hndlr.load_images(path + "/dark/")
    Y = img_hndlr.load_images(path + "/true/")

    X = img_hndlr.preprocess_images(X)
    Y = img_hndlr.preprocess_images(Y)
    print("XY shapes", X.shape, Y.shape)

    print("python msrnet.py")
    m=None
    xx = Input((64, 64, 3))  # tf.constant(np.ndarray((?,100,100,3),dtype=np.float32))
    # nnOut,loss=nn(xx)
    nnOut = nn(xx,actFunc)

    m = Model(xx, nnOut)
    m.compile(optimizer="adam", loss='mean_squared_error', metrics=['mean_squared_error'])

    if os.path.isfile(MODEL_FILE_NAME):
        m.load_weights(MODEL_FILE_NAME)
    else:

        print(m.summary())

        m.fit(X,Y,verbose=1,epochs=100,shuffle=True)
        # for e in range(20):
        #     goOn=input("Go on? y or n :")
        #     if goOn=='n':
        #         break

        m.save_weights(MODEL_FILE_NAME)

        # m.save(MODEL_FILE_NAME)

    yPred=m.predict(X)

    # yPred = yPred - np.min(yPred)
    # yPred = yPred / np.max(yPred + EPS)
    yPred=img_hndlr.inv_preprocess_images(yPred)
    yPred=yPred.astype(np.uint8)



    x1LayerOut=Model(xx,m.get_layer("X1").output).predict(X,verbose=1)
    x1LayerOut=x1LayerOut-np.min(x1LayerOut)
    x1LayerOut=x1LayerOut/np.max(x1LayerOut+EPS)
    x1LayerOut=img_hndlr.inv_preprocess_images(x1LayerOut).astype(np.uint8)
    x2LayerOut=Model(xx,m.get_layer("X2").output).predict(X,verbose=1)
    x2LayerOut=x2LayerOut-np.min(x2LayerOut)
    x2LayerOut=x2LayerOut/np.max(x2LayerOut+EPS)
    x2LayerOut=img_hndlr.inv_preprocess_images(x2LayerOut).astype(np.uint8)
    diffLayerOut=Model(xx,m.get_layer("DIFF").output).predict(X,verbose=1)
    diffLayerOut=diffLayerOut-np.min(diffLayerOut)
    diffLayerOut=diffLayerOut/np.max(diffLayerOut+EPS)
    diffLayerOut=img_hndlr.inv_preprocess_images(diffLayerOut).astype(np.uint8)


    X=img_hndlr.inv_preprocess_images(X).astype(np.uint8)
    Y=img_hndlr.inv_preprocess_images(Y).astype(np.uint8)


    print("yPred shape",yPred.shape,"max",np.max(yPred),"min",np.min(yPred))

    imgs=np.concatenate((X,Y,yPred,x1LayerOut,x2LayerOut,diffLayerOut),axis=1)
    imgs=splitInto4(imgs)
    img_hndlr.save_images(path + "/output-detailed/", imgs)
    img_hndlr.save_images(path + "/output/", yPred)

    np.savez(npzSaveFile,yTrue=Y,yPred=yPred)
    print("NPZ saved {}".format(npzSaveFile))
