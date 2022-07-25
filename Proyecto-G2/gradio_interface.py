import gradio as gr
from sklearn.preprocessing import Normalizer
import pickle

def predictor(text):
    count_vector = pickle.load(open('salida/naive_bayes.vector', 'rb'))
    naive_bayes = pickle.load(open('salida/naive_bayes.model', 'rb'))
    data = count_vector.transform([text])
    norm = Normalizer().fit_transform(data)
    return 'Hate Speech' if naive_bayes.predict(norm)[0] == 1 else 'Not Hate Speech'

iface = gr.Interface(fn=predictor, inputs="text", outputs="text")
iface.launch()