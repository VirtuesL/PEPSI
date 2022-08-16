import tensorflow as tf
import tensorflow_datasets as tfds
from model import PEPSI # pepsi model
from operations import epochCallback, mask_dataset # loss functions
import glob

celeba = tfds.load("celeb_a", split="train", batch_size=8)
dataset = celeba.map(mask_dataset)
num_ims = len(dataset)

fw = tf.summary.create_file_writer("logs")

#training loop
tbcb = tf.keras.callbacks.TensorBoard()
epcb = epochCallback()
EPOCHS = 100
loops = 16
model = PEPSI(max_iters=8*num_ims*loops)
model.compile(
  d_optimiser=tf.optimizers.Adam(learning_rate=0.0004, beta_1=0.5, beta_2=0.9),
  g_optimiser=tf.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
  )

data = dataset.shuffle(100).take(1)
masked, real, mask = list(*data.as_numpy_iterator())

saves = glob.glob("*.h5")
if saves:
  latest_save = (sorted(saves, key=lambda a: int(a.split('_')[-1].split('.')[0]))[-1],len(saves))
  model((masked,real,mask))
  print("loading newest save %d"%latest_save[0])
  model.load_weights("pepsi_weights_%d.h5"%latest_save[0],by_name=True)
else:
  print("no new saves")
  latest_save = (None, 0)

for i in range(latest_save[1],loops):
  model.fit(dataset.repeat(),batch_size=8,callbacks=[tbcb,epcb],steps_per_epoch=num_ims//(EPOCHS),epochs=EPOCHS)
  I_co, I_ge, image_result, D_fake_red, D_real_red  = model.predict((masked,real,mask),batch_size=8)
  with fw.as_default():
    tf.summary.image("inputs",masked,step=i,max_outputs=8)
    tf.summary.image("image_result",image_result,step=i,max_outputs=8)
    tf.summary.image("reals",real,step=i,max_outputs=8)
    tf.summary.image("masks",mask,step=i,max_outputs=8)
    tf.summary.image("I_ge",I_ge,step=i,max_outputs=8)
    tf.summary.image("I_ge",I_co,step=i,max_outputs=8)
  model.save_weights("pepsi_weights_%d.h5"%i)
    
