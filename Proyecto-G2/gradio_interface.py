import gradio as gr
import pickle
from gradio.components import Dropdown, Textbox
from sklearn.preprocessing import Normalizer

def predictor(model, text):
    if model == "Naive Bayes":
        count_vector = pickle.load(open('salida/naive_bayes.vector', 'rb'))
        naive_bayes = pickle.load(open('salida/naive_bayes.model', 'rb'))
        data = count_vector.transform([text])
        norm = Normalizer().fit_transform(data)
        response = 'Hate Speech' if naive_bayes.predict(norm)[0] == 1 else 'Not Hate Speech'
    else:
        response = "Modelo no disponible"
    return response

input1 = Dropdown(["Naive Bayes", "Logistic Regression", "Transformers"], label="Model")
input2 = Textbox(placeholder="Enter Phrase", label="Phrase")
output = Textbox(label="Result")

iface = gr.Interface(fn=predictor, inputs=[input1, input2], outputs=[output], live=True,)
iface.launch()