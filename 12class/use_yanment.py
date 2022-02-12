## Updated CHH
import tensorflow as tf
import y_params as yamnet_params
import yamnet as yamnet_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers as L
from tensorflow.keras import optimizers

class ReduceMeanLayer(L.Layer):
  def __init__(self, axis=0, **kwargs):
    super(ReduceMeanLayer, self).__init__(**kwargs)
    self.axis = axis

  def call(self, input):
    return tf.math.reduce_mean(input, axis=self.axis)


def Yemnet_Model(nCategories):

    params = yamnet_params.Params(sample_rate=16000, patch_hop_seconds=0.1)
    # Set up the YAMNet model.
    class_names = yamnet_model.class_names('data/yamnet_class_map.csv')
    yamnet_layer = yamnet_model.yamnet_frames_model(params)
    yamnet_layer.load_weights('data/yamnet.h5')
    ## Define model 

    inputs = L.Input(shape=(), dtype=tf.float32, name='audio')
    _, embeddings_output, _ = yamnet_layer(inputs) # predictions, embeddings_output, log_mel_spectrogram
    serving_outputs = L.Dense(nCategories)(embeddings_output)
    outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)

    model = Model(inputs=[inputs], outputs = [outputs], name = "Yamnet_pre")

    return model
