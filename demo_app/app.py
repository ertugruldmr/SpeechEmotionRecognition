import librosa
import numpy as np
import tensorflow as tf
import gradio as gr


# File Paths
model_path = "sound_emotion_rec_model"
categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
model = tf.keras.models.load_model(model_path)


# loading the files
def extract_mfcc(audio_path, duration=3, offset=0.5, n_mfcc=40):
    # loading the data
    y, sr = librosa.load(audio_path, duration=duration, offset=offset)
    
    # extracting the voice feature
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    
    return mfcc

def prepare_data(audio_path):
  
  # extracting the features
  features = extract_mfcc(audio_path)

  # adjusting the shape
  features = [x for x in features]
  features = np.array(features)
  features = np.expand_dims(features, -1)

  return features

def clsf(audio_path):
  
  # extracting the features
  features = prepare_data(audio_path)

  # batching the data
  sample = np.expand_dims(features, axis=0)

  # predicting
  preds = model.predict(sample)[0]

  # results
  confidences = {categories[i]:np.round(float(preds[i]), 3) for i in range(len(categories))}

  return confidences

def pre_processor(audio_path):

  # load the audio file
  x, sample_rate = librosa.load(audio_path)
  
  # feature extracting (mfccs is an aduio feature)
  mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)
  feature = mfccs
  
  return feature



# GUI Component
gui_params = {
    "fn":clsf, 
    "inputs":gr.Audio(source="upload", type="filepath"),
    "outputs" : "label",
    #live=True,
    "examples" : "examples"
    
} 
demo = gr.Interface(**gui_params)

# Launching the demo
if __name__ == "__main__":
    demo.launch()
