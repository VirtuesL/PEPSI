import tensorflow as tf
import numpy as np
import random as rr
import math as mt
import imageio as misc
import cv2


def spectral_norm(w,u):
  w_shape = w.shape.as_list()
  w = tf.reshape(w,[-1,w_shape[-1]])
  v_ = tf.matmul(u, tf.transpose(w))
  v_hat = tf.nn.l2_normalize(v_)

  u_ = tf.matmul(v_hat, w)
  u_hat = tf.nn.l2_normalize(u_)

  sigma = tf.matmul(tf.matmul(v_hat,w),tf.transpose(u_hat))
  return w / sigma


def ff_mask(size, b_zise, maxLen, maxWid, maxAng, maxNum, maxVer, minLen = 20, minWid = 15, minVer = 5):

    mask = np.ones((b_zise, size, size, 3))

    num = rr.randint(3, maxNum)

    for i in range(num):
        startX = rr.randint(0, size)
        startY = rr.randint(0, size)
        numVer = rr.randint(minVer, maxVer)
        width = rr.randint(minWid, maxWid)
        for j in range(numVer):
            angle = rr.uniform(-maxAng, maxAng)
            length = rr.randint(minLen, maxLen)

            endX = min(size-1, max(0, int(startX + length * mt.sin(angle))))
            endY = min(size-1, max(0, int(startY + length * mt.cos(angle))))

            if endX >= startX:
                lowx = startX
                highx = endX
            else:
                lowx = endX
                highx = startX
            if endY >= startY:
                lowy = startY
                highy = endY
            else:
                lowy = endY
                highy = startY

            if abs(startY-endY) + abs(startX - endX) != 0:

                wlx = max(0, lowx-int(abs(width * mt.cos(angle))))
                whx = min(size - 1,  highx+1 + int(abs(width * mt.cos(angle))))
                wly = max(0, lowy - int(abs(width * mt.sin(angle))))
                why = min(size - 1, highy+1 + int(abs(width * mt.sin(angle))))

                for x in range(wlx, whx):
                    for y in range(wly, why):

                        d = abs((endY-startY)*x - (endX -startX)*y - endY*startX + startY*endX) / mt.sqrt((startY-endY)**2 + (startX -endX)**2)

                        if d <= width:
                            mask[:, x, y, :] = 0

            wlx = max(0, lowx-width)
            whx = min(size - 1, highx+width+1)
            wly = max(0, lowy - width)
            why = min(size - 1, highy + width + 1)

            for x2 in range(wlx, whx):
                for y2 in range(wly, why):

                    d1 = (startX - x2) ** 2 + (startY - y2) ** 2
                    d2 = (endX - x2) ** 2 + (endY - y2) ** 2

                    if np.sqrt(d1) <= width:
                        mask[:, x2, y2, :] = 0
                    if np.sqrt(d2) <= width:
                        mask[:, x2, y2, :] = 0
            startX = endX
            startY = endY

    return mask

def mask_dataset(inputs):
    res = tf.image.resize_with_crop_or_pad(inputs['image'],171,171)
    res = tf.image.resize(inputs['image'],(256,256))
    data = tf.cast(res,tf.float32)
    mask = ff_mask_batch(256,8,50, 30, 3.14, 5, 15)
    reals = data/255.0 *2.0 -1.0
    data_m = (data*mask)/255.0 *2.0 - 1.0
    return (data_m,reals,mask)

def ff_mask_batch(size, b_size, maxLen, maxWid, maxAng, maxNum, maxVer, minLen = 20, minWid = 15, minVer = 5):
    mask = None
    temp = ff_mask(size, 1, maxLen, maxWid, maxAng, maxNum, maxVer, minLen=minLen, minWid=minWid, minVer=minVer)
    temp = temp[0]
    mask = np.expand_dims(temp, 0)
    for ib in range(1,b_size):
        mask = np.concatenate((mask, np.expand_dims(temp, 0)), 0)
        temp = np.rot90(temp)
        if ib % 3 == 0:
            temp = np.fliplr(temp)

    return mask

def MakeImageBlock(Qfilenames, Height, Width, i, batch_size, resize=True):

    iCount = 0
    Image = np.zeros((batch_size, Height, Width, 3))

    #Query Image block

    for iL in range((i * batch_size), (i * batch_size) + batch_size):

        Loadimage = misc.imread(Qfilenames[iL])

        #if Gray make it colors
        if Loadimage.ndim == 2:
            Loadimage = np.expand_dims(Loadimage, 2)
            Loadimage = np.tile(Loadimage, (1, 1, 3))

        if Loadimage.shape[2] != 3:
            Loadimage = Loadimage[:, :, 0:3]

        if resize:
            Loadimage = cv2.resize(Loadimage,(256,256))

        Loadimage = Loadimage.astype(np.float32)

        # Mean Value subtraction
        Loadimage = (Loadimage / 255.0 - 0.5) * 2

        Image[iCount] = np.array(Loadimage, dtype=float)
        iCount = iCount + 1

    return Image
