import tensorflow as tf
import tensorflow_datasets as tfds
from model import PEPSI # pepsi model
from operations import epochCallback, mask_dataset # loss functions

#training loop
tbcb = tf.keras.callbacks.TensorBoard()
epcb = epochCallback()
EPOCHS = 100
model = PEPSI(max_epochs=EPOCHS)
model.compile(
  d_optimiser=tf.optimizers.Adam(learning_rate=0.0004, beta_1=0.5, beta_2=0.9),
  g_optimiser=tf.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
  )
dataset:tf.data.Dataset = tfds.ImageFolder("F:\\datasets").as_dataset("train",batch_size=8,shuffle_files=True)
fw = tf.summary.create_file_writer("logs")
dataset = dataset.map(mask_dataset)

data = dataset.take(1)
masked, real, mask = list(*data.as_numpy_iterator())

for i in range(1,16):
  model.fit(dataset.repeat(),batch_size=8,callbacks=[tbcb,epcb],steps_per_epoch=202599//(8*EPOCHS),epochs=EPOCHS)
  predicts = model.predict((masked,real,mask),batch_size=8)
  I_co, I_ge, image_result, D_fake_red, D_real_red = predicts
  with fw.as_default():
    tf.summary.image("inputs",masked,step=i,max_outputs=8)
    tf.summary.image("outputs",image_result,step=i,max_outputs=8)
    tf.summary.image("reals",real,step=i,max_outputs=8)
    tf.summary.image("masks",mask,step=i,max_outputs=8)
  model.save_weights("pepsi_weights_%d.h5"%i)
    
