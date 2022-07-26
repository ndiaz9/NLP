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
    elif model == "Logistic Regression":
        count_vector = pickle.load(open('salida/logistic_regression.vector', 'rb'))
        logistic_regression = pickle.load(open('salida/logistic_regression.model', 'rb'))
        data = count_vector.transform([text])
        norm = Normalizer().fit_transform(data)
        response = 'Hate Speech' if logistic_regression.predict(norm)[0] == 1 else 'Not Hate Speech'
    elif model == "Random Forest":
        count_vector = pickle.load(open('salida/random_forest.vector', 'rb'))
        random_forest = pickle.load(open('salida/random_forest.model', 'rb'))
        data = count_vector.transform([text])
        norm = Normalizer().fit_transform(data)
        response = 'Hate Speech' if random_forest.predict(norm)[0] == 1 else 'Not Hate Speech'
    else:
        response = "Modelo no disponible"
    return response

input1 = Dropdown(["Naive Bayes", "Logistic Regression","Random Forest", "Transformers"], label="Model")
input2 = Textbox(placeholder="Enter Phrase", label="Phrase")
output = Textbox(label="Result")

iface = gr.Interface(fn=predictor, inputs=[input1, input2], outputs=[output], live=True,)
iface.launch()