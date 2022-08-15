import tensorflow as tf
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras import layers
from operations import *

class Encoder(layers.Layer):
  def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
    super(Encoder,self).__init__(trainable, name, dtype, dynamic, **kwargs)
    
    self.CL1 = layers.Conv2D(32,[5,5],strides=(1,1),activation=tf.nn.elu,input_shape=(256, 256, 6))
    self.CL2 = layers.Conv2D(64,[3,3],strides=(2,2),activation=tf.nn.elu)
    self.CL3 = layers.Conv2D(64,[3,3],strides=(1,1),activation=tf.nn.elu)
    self.CL4 = layers.Conv2D(128,[3,3],strides=(2,2),activation=tf.nn.elu)
    self.CL5 = layers.Conv2D(128,[3,3],strides=(1,1),activation=tf.nn.elu)
    self.CL6 = layers.Conv2D(256,[3,3],strides=(2,2),activation=tf.nn.elu)
    
    self.DCL1 = layers.Conv2D(256,[3,3],strides=(1,1),dilation_rate=2,activation=tf.nn.elu)
    self.DCL2 = layers.Conv2D(256,[3,3],strides=(1,1),dilation_rate=4,activation=tf.nn.elu)
    self.DCL3 = layers.Conv2D(256,[3,3],strides=(1,1),dilation_rate=8,activation=tf.nn.elu)
    self.DCL4 = layers.Conv2D(256,[3,3],strides=(1,1),dilation_rate=16,activation=tf.nn.elu)

  def call(self, inputs:tf.Tensor):
    pad  = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT")
    CL1  = self.CL1(pad)
    CLP1 = tf.pad(CL1, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
    CL2  = self.CL2(CLP1)
    CLP2 = tf.pad(CL2, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
    CL3 = self.CL3(CLP2)
    CLP3 = tf.pad(CL3, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
    CL4 = self.CL4(CLP3)
    CLP4 = tf.pad(CL4, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
    CL5 = self.CL5(CLP4)
    CLP5 = tf.pad(CL5, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
    CL6 = self.CL6(CLP5)
    CLP6 = tf.pad(CL6, [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT")
    DCL1 = self.DCL1(CLP6)
    DCLP1 = tf.pad(DCL1, [[0, 0], [4, 4], [4, 4], [0, 0]], "REFLECT")
    DCL2 = self.DCL2(DCLP1)
    DCLP2 = tf.pad(DCL2, [[0, 0], [8, 8], [8, 8], [0, 0]], "REFLECT")
    DCL3 = self.DCL3(DCLP2)
    DCLP3 = tf.pad(DCL3, [[0, 0], [16, 16], [16, 16], [0, 0]], "REFLECT")
    DCLP4 = self.DCL4(DCLP3)
    
    return DCLP4
  
class Decoder(layers.Layer):
  class nnConv2D(layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, dims=(128,128), resize=(64,64), **kwargs):
      super().__init__(trainable, name, dtype, dynamic, **kwargs)
      self.L1 = layers.Conv2D(dims[0],[3,3],strides=(1,1),activation=tf.nn.elu)
      self.L2 = layers.Conv2D(dims[1],[3,3],strides=(1,1),activation=tf.nn.elu)
      self.resize = resize

    def call(self, inputs):
      prepad = tf.pad(inputs, [[0,0], [1,1], [1,1],[0,0]], "REFLECT")
      L1 = self.L1(prepad)
      LP1 = tf.pad(L1, [[0,0], [1,1], [1,1],[0,0]], "REFLECT")
      L2 = self.L2(LP1)
      LP2 = tf.pad(L2, [[0,0], [1,1], [1,1],[0,0]], "REFLECT")
      return tf.image.resize(LP2,self.resize,tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
    super().__init__(trainable, name, dtype, dynamic, **kwargs)
    self.DL1 = self.nnConv2D(dims=(128,128),resize=(64,64)) # 64 64 128
    self.DL2 = self.nnConv2D(dims=(64,64),resize=(128,128)) 
    self.DL3 = self.nnConv2D(dims=(32,32),resize=(256,256))
    self.DL4 = self.nnConv2D(dims=(16,16),resize=(256,256))
    self.LLO = layers.Conv2D(3,[3,3],strides=(1,1),padding="same")


  def call(self, inputs):
    DLP1 = self.DL1(inputs)
    DLP2 = self.DL2(DLP1)
    DLP3 = self.DL3(DLP2)
    DLP4 = self.DL4(DLP3)
    LLO = self.LLO(DLP4)
    return tf.clip_by_value(LLO, -1.0, 1.0)

class Discriminator_red(layers.Layer):
  def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
    super().__init__(trainable, name, dtype, dynamic, **kwargs)
    self.L1 = SpectralNormalization(layers.Conv2D(64 ,(5,5),strides=(2,2),activation=tf.nn.leaky_relu))
    self.L2 = SpectralNormalization(layers.Conv2D(128,(5,5),strides=(2,2),activation=tf.nn.leaky_relu))
    self.L3 = SpectralNormalization(layers.Conv2D(256,(5,5),strides=(2,2),activation=tf.nn.leaky_relu))
    self.L4 = SpectralNormalization(layers.Conv2D(256,(5,5),strides=(2,2),activation=tf.nn.leaky_relu))
    self.L5 = SpectralNormalization(layers.Conv2D(256,(5,5),strides=(2,2),activation=tf.nn.leaky_relu))
    self.L6 = SpectralNormalization(layers.Conv2D(256,(5,5),strides=(2,2),activation=tf.nn.leaky_relu))
    self.L7 = SpectralNormalization(layers.Dense(1))

  def call(self, inputs):
    L1 = self.L1(inputs)
    L2 = self.L2(L1)
    L3 = self.L3(L2)
    L4 = self.L4(L3)
    L5 = self.L5(L4)
    L6 = self.L6(L5)
    FC = self.L7(L6)
    return FC

class ContextAwareModule(layers.Layer):
  def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
    super().__init__(trainable, name, dtype, dynamic, **kwargs)

  def build(self, input_shape):
    _, _, _, dims = input_shape[0].as_list()
    self.ACL2 = layers.Conv2D(dims, [1,1], strides=(1,1), activation=tf.nn.elu)

  def call(self, inputs):
    g_in, mask = inputs
    b, h, w, dims = g_in.shape.as_list()
    b = b or 8
    temp = tf.image.resize(mask,(h,w),tf.image.ResizeMethod.NEAREST_NEIGHBOR) # b 128 128 1
    temp = tf.expand_dims(temp[:, :, :, 0], 3) # b 128 128 128
    
    mask_r = tf.tile(temp, [1, 1, 1, dims])
    bg = g_in *mask_r

    kn = (3-1)//2
    c = (h-2*kn)*(w-2*kn)

    patch1 = tf.image.extract_patches(bg,[1,3,3,1],[1,1,1,1],[1,1,1,1],'VALID')
    patch1 = tf.reshape(patch1, [b, 1, c, 3*3*dims])
    patch1 = tf.reshape(patch1, (b, 1, 1, c, 3*3*dims))
    patch1 = tf.transpose(patch1, [0, 1, 2, 4, 3])
    
    patch2 = tf.image.extract_patches(g_in,[1,3,3,1],[1,1,1,1],[1,1,1,1],'SAME')
    ACL = []

    for i in range(b):
      k1 = patch1[i, :, :, :, :]
      k1d = tf.reduce_sum(tf.square(k1),axis=2)
      k2 = tf.reshape(k1, (3, 3, dims, c))
      ww = patch2[i, :, :, :]
      wwd = tf.reduce_sum(tf.square(ww), axis=2, keepdims=True)
      ft = tf.expand_dims(ww, 0)

      CS = tf.nn.conv2d(ft, k1, strides=[1,1,1,1],padding='SAME')
      tt = k1d + wwd

      DS1 = tf.expand_dims(tt,0) - 2*CS
      DS2 = (DS1 - tf.reduce_mean(DS1, 3, True)) / tf.math.reduce_std(DS1,3,True)
      DS2 = -tf.tanh(DS2)

      CA = tf.math.softmax(50.0*DS2)
      ACLt = tf.nn.conv2d_transpose(CA,k2,output_shape=[1,h,w,dims], strides=[1,1,1,1], padding='SAME') / 9
      if i == 0:
        ACL = ACLt
      else:
        ACL = tf.concat([ACL,ACLt],0)

    ACL = bg + ACL * (1-mask_r)
    con1 = tf.concat([g_in,ACL], 3)
    return self.ACL2(con1)

class PEPSI(tf.keras.Model):
  def __init__(self, max_epochs):
    super(PEPSI, self).__init__()
    self.max_epochs = max_epochs
    self.alpha = 0
    self.encoder = Encoder(name="G_en")
    self.decoder = Decoder(name="G_de")
    self.cam = ContextAwareModule(name="CB1")
    self.RED = Discriminator_red(name="disc_red")

  def set_alpha(self,alpha):
    self.alpha = alpha

  def Loss_D(self, D_real_red, D_fake_red):
    return tf.reduce_mean(tf.nn.relu(1+D_fake_red)) + tf.reduce_mean(tf.nn.relu(1-D_real_red))

  
  def Loss_G(self, I_co, I_ge, D_fake_red, Y):
    Loss_gan = -tf.reduce_mean(D_fake_red)

    Loss_s_re = tf.reduce_mean(tf.abs(I_ge - Y))
    Loss_hat = tf.reduce_mean(tf.abs(I_co - Y))

    return (0.1*Loss_gan + 10*Loss_s_re + 5*(1-self.alpha) * Loss_hat, Loss_s_re)

    
  def compile(self, d_optimiser, g_optimiser):
    super(PEPSI, self).compile()
    self.d_optimiser = d_optimiser
    self.g_optimiser = g_optimiser
    self.d_loss_metric = tf.keras.metrics.Mean("d_loss")
    self.g_loss_metric = tf.keras.metrics.Mean("g_loss")

  @property
  def metrics(self):
    return [self.d_loss_metric, self.g_loss_metric]

  def call(self, inputs):
    inputs, reals, mask = inputs
    encoded = self.encoder(tf.concat([inputs, mask], 3))
    cammed = self.cam((encoded,mask))
    I_co = self.decoder(encoded)
    I_ge = self.decoder(cammed)

    image_result = I_ge * (1-mask) + reals*mask

    D_real_red = self.RED(reals)
    D_fake_red = self.RED(image_result)

    return I_co, I_ge, image_result, D_fake_red, D_real_red

  def train_step(self, data:tf.Tensor):
    masked, reals, masks = data
    with tf.GradientTape(persistent=True) as tape:
      I_co, I_ge, image_result, D_fake_red, D_real_red = self(data)
      loss_d = self.Loss_D(D_real_red, D_fake_red)
      loss_g, loss_s_re = self.Loss_G(I_co, I_ge,D_fake_red,reals)
    
    self.d_loss_metric.update_state(loss_d)
    self.g_loss_metric.update_state(loss_g)

    grads = tape.gradient(loss_d,self.RED.trainable_weights)
    self.d_optimiser.apply_gradients(zip(grads,self.RED.trainable_weights))
    grads = tape.gradient(loss_g, self.encoder.trainable_weights + self.decoder.trainable_weights + self.cam.trainable_weights)
    self.g_optimiser.apply_gradients(zip(grads,self.encoder.trainable_weights + self.decoder.trainable_weights + self.cam.trainable_weights))
  

    return { "d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result() }