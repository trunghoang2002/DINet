import numpy as np
import warnings
import resampy
from scipy.io import wavfile
from python_speech_features import mfcc
import tensorflow as tf
import time

# tf.compat.v1.config.optimizer.set_jit(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if len(gpus) > 1:
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[1], True)

class DeepSpeech():
    def __init__(self,model_path):
        # self.graph, self.logits_ph, self.input_node_ph, self.input_lengths_ph \
        #     = self._prepare_deepspeech_net(model_path)
        # self.target_sample_rate = 16000
        # self.sess = tf.compat.v1.Session(graph=self.graph)
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.io.gfile.GFile(model_path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="deepspeech")
            self.logits_ph = self.graph.get_tensor_by_name("deepspeech/logits:0")
            print("self.logits_ph", self.logits_ph)
            self.input_node_ph = self.graph.get_tensor_by_name("deepspeech/input_node:0")
            print("self.input_node_ph", self.input_node_ph)
            self.input_lengths_ph = self.graph.get_tensor_by_name("deepspeech/input_lengths:0")
            print("self.input_lengths_ph", self.input_lengths_ph)
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.5
            config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
            config.log_device_placement = True
            self.session = tf.compat.v1.Session(config=config)
        self.target_sample_rate = 16000

    def _prepare_deepspeech_net(self,deepspeech_pb_path):
        tf.compat.v1.disable_eager_execution()  # Bật chế độ tương thích TF1.x
        with tf.io.gfile.GFile(deepspeech_pb_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        graph = tf.compat.v1.get_default_graph()
        tf.import_graph_def(graph_def, name="deepspeech")
        logits_ph = graph.get_tensor_by_name("deepspeech/logits:0")
        input_node_ph = graph.get_tensor_by_name("deepspeech/input_node:0")
        input_lengths_ph = graph.get_tensor_by_name("deepspeech/input_lengths:0")

        return graph, logits_ph, input_node_ph, input_lengths_ph

    def conv_audio_to_deepspeech_input_vector(self,audio,
                                              sample_rate,
                                              num_cepstrum,
                                              num_context):
        # Get mfcc coefficients:
        features = mfcc(
            signal=audio,
            samplerate=sample_rate,
            numcep=num_cepstrum)

        # We only keep every second feature (BiRNN stride = 2):
        features = features[::2]

        # One stride per time step in the input:
        num_strides = len(features)

        # Add empty initial and final contexts:
        empty_context = np.zeros((num_context, num_cepstrum), dtype=features.dtype)
        features = np.concatenate((empty_context, features, empty_context))

        # Create a view into the array with overlapping strides of size
        # numcontext (past) + 1 (present) + numcontext (future):
        window_size = 2 * num_context + 1
        train_inputs = np.lib.stride_tricks.as_strided(
            features,
            shape=(num_strides, window_size, num_cepstrum),
            strides=(features.strides[0],
                     features.strides[0], features.strides[1]),
            writeable=False)

        # Flatten the second and third dimensions:
        train_inputs = np.reshape(train_inputs, [num_strides, -1])

        train_inputs = np.copy(train_inputs)
        train_inputs = (train_inputs - np.mean(train_inputs)) / \
                       np.std(train_inputs)

        return train_inputs

    def compute_audio_feature(self, audio_path):
        start_time = time.time()
        audio_sample_rate, audio = wavfile.read(audio_path)
        print(audio_sample_rate)
        if audio.ndim != 1:
            warnings.warn(
                "Audio has multiple channels, the first channel is used")
            audio = audio[:, 0]
        if audio_sample_rate != self.target_sample_rate:
            resampled_audio = resampy.resample(
                x=audio.astype(float),
                sr_orig=audio_sample_rate,
                sr_new=self.target_sample_rate)
        else:
            # resampled_audio = audio.astype(np.float)
            resampled_audio = audio.astype(float)
        end_time = time.time()
        print(f"Time taken to resample audio: {end_time - start_time} seconds")

        start_time = time.time()
        input_vector = self.conv_audio_to_deepspeech_input_vector(
            audio=resampled_audio.astype(np.int16),
            sample_rate=self.target_sample_rate,
            num_cepstrum=26,
            num_context=9)
        end_time = time.time()
        print("input_vector", input_vector.shape)
        print(f"Time taken to convert audio to input vector: {end_time - start_time} seconds")
        start_time = time.time()
        
        network_output = self.session.run(
                self.logits_ph,
                feed_dict={
                    self.input_node_ph: input_vector[np.newaxis, ...],
                    self.input_lengths_ph: [input_vector.shape[0]]
                })
        print("network_output", network_output.shape)
        end_time = time.time()
        print(f"Time taken to run network: {end_time - start_time} seconds")
        ds_features = network_output[::2,0,:]
        print("ds_features", ds_features.shape)
        return ds_features
    
    def compute_audio_feature_from_data(self, audio_array, samplerate):
        start_time = time.time()
        if audio_array.ndim != 1:
            warnings.warn(
                "Audio has multiple channels, the first channel is used")
            audio_array = audio_array[:, 0]
        if samplerate != self.target_sample_rate:
            resampled_audio = resampy.resample(
                x=audio_array.astype(float),
                sr_orig=samplerate,
                sr_new=self.target_sample_rate)
        else:
            resampled_audio = audio_array.astype(float)
        end_time = time.time()
        print(f"Time taken to resample audio: {end_time - start_time} seconds")

        start_time = time.time()
        input_vector = self.conv_audio_to_deepspeech_input_vector(
            audio=resampled_audio.astype(np.int16),
            sample_rate=self.target_sample_rate,
            num_cepstrum=26,
            num_context=9)
        end_time = time.time()
        print("input_vector", input_vector.shape)
        print(f"Time taken to convert audio to input vector: {end_time - start_time} seconds")
        start_time = time.time()
        network_output = self.session.run(
                self.logits_ph,
                feed_dict={
                    self.input_node_ph: input_vector[np.newaxis, ...],
                    self.input_lengths_ph: [input_vector.shape[0]]
                })
        print("network_output", network_output.shape)
        end_time = time.time()
        print(f"Time taken to run network: {end_time - start_time} seconds")
        ds_features = network_output[::2,0,:]
        print("ds_features", ds_features.shape)
        return ds_features
    
if __name__ == '__main__':
    audio_path = r'./00168.wav'
    model_path = r'./output_graph.pb'
    DSModel = DeepSpeech(model_path)
    ds_feature = DSModel.compute_audio_feature(audio_path)
    print(ds_feature)

# input_vector.shape = (410, 494) = (num_strides(=410), window_size(=2*num_context+1=2*8+1=19) * num_cepstrum(=26) )
# network_output.shape = (410, 1, 29) = (num_strides(=410), batch_size(=1), num_classes(=29))