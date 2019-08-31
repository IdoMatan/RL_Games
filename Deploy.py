import tensorflow as tf
keras = tf.keras
from EdgeDevice import *
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = tf.keras.models.load_model('./trained_model')

episode = deploy(model, camID=0)

