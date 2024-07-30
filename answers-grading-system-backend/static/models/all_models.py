import pickle
import numpy as np
from nltk.tokenize import word_tokenize
import warnings
import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from graphene import ObjectType, String, Schema
from pydantic import BaseModel
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow
import numpy as np
import tensorflow_addons as tfa
from keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings("ignore")
nltk.download("punkt")
app = FastAPI()
nltk.download('punkt')
nltk.download('wordnet')

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def get_word_vector(tokens, model_word2vec):
    textvector = np.zeros((100,), dtype="float32")
    for token in tokens:
        try:
            textvector += model_word2vec.wv[token]
        except KeyError:
            continue
    return textvector


def preprocces_input(text, model_word2vec):
    text = text.lower()
    tokens = word_tokenize(text)
    textvector = get_word_vector(tokens, model_word2vec)
    return textvector
lemmer = WordNetLemmatizer()
tokenizer = Tokenizer(filters=''''!"#$%&()+,-./:;<=>?@[\\]^{|}~\t\n÷×؛<>()&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''''')
list=[0,28,29,226,10,11,14,9,18,8,7]
def predict(question_id, input):
    model_path = (
        "./static/models/trained_models/LSTM_Question" + str(question_id) + ".h5"
    )
    with custom_object_scope({'f1_m': f1_m, 'Addons>CohenKappa': tfa.metrics.CohenKappa}):
        model = load_model(model_path)
    newsentence = nltk.word_tokenize(input)
    sequence_test = pad_sequences(tokenizer.texts_to_sequences([newsentence]), maxlen=list[question_id])
    prediction = model.predict(sequence_test)
    predicted_class_index = np.argmax(prediction)
    index_to_class = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}
    predicted_class = index_to_class[predicted_class_index]
    return predicted_class






