import gradio as gr
import pickle
from gradio.components import Dropdown, Textbox
from sklearn.preprocessing import Normalizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag, download

download('wordnet')
download('sentiwordnet')
download('omw-1.4')
download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to Wordnet tags
    """
    return {'J': wn.ADJ, 'N': wn.NOUN, 'R': wn.ADV, 'V': wn.VERB}.get(tag[0])


def get_sentiment(word, tag):
    """ 
    returns list of pos neg and objective score. But returns empty list if not present in senti wordnet. 
    """
    wn_tag = penn_to_wn(tag)
    valid_wn_tags = (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB)
    if wn_tag not in valid_wn_tags: return (0.0, 0.0, 1.0)
    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma: return (0.0, 0.0, 1.0)
    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets: return (0.0, 0.0, 1.0)
    swn_synset = swn.senti_synset(synsets[0].name())
    return (swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score())


def lexicon_sentiwordnet(document: str) -> list:
    """
    return the lexicon features between others:
    - sum of positive, negative and objetive scores
    - mean of the sum of positive, negative and objetive scores
    - cant of words with positive, negative and objetive scores over 0.5
    """
    neg_scores, pos_scores, obj_scores= [], [], []
    words = word_tokenize(document)
    cant_words = len(words)
    pos_words = pos_tag(words)
    for word, tag in pos_words:
        scores = get_sentiment(word, tag)
        pos_scores.append(scores[0])
        neg_scores.append(scores[1])
        obj_scores.append(scores[2])
    pos_score, neg_score, obj_score = sum(pos_scores), sum(neg_scores), sum(obj_scores)
    pond_pos, pond_neg, pond_obj = pos_score/cant_words, neg_score/cant_words, obj_score/cant_words
    cant_pos, cant_neg, cant_obj = len([item for item in pos_scores if item >= 0.5]), len([item for item in neg_scores if item >= 0.5]), len([item for item in obj_scores if item >= 0.5])
    most_important = 1 if neg_score > pos_score else 0
    return [
        pos_score, neg_score, obj_score, pond_pos, pond_neg, pond_obj, 
        cant_pos/cant_words, cant_neg/cant_words, cant_obj/cant_words, 
        cant_pos, cant_neg, cant_obj, most_important,
    ]


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
    elif model == "SentiWordNet Lexicon":
        naive_bayes = pickle.load(open('salida/swn_lexicon_naive_bayes.model', 'rb'))
        data = lexicon_sentiwordnet(text) if text else [0,0,0,0,0,0,0,0,0,0,0,0,0]
        response = 'Hate Speech' if naive_bayes.predict([data])[0] == 1 and text != '' else 'Not Hate Speech'
    elif model == "Random Forest":
        count_vector = pickle.load(open('salida/random_forest.vector', 'rb'))
        random_forest = pickle.load(open('salida/random_forest.model', 'rb'))
        data = count_vector.transform([text])
        norm = Normalizer().fit_transform(data)
        response = 'Hate Speech' if random_forest.predict(norm)[0] == 1 else 'Not Hate Speech'
    else:
        response = "Modelo no disponible"
    return response

input1 = Dropdown(["Naive Bayes", "Logistic Regression", "SentiWordNet Lexicon", "Random Forest", "Transformers"], label="Model")
input2 = Textbox(placeholder="Enter Phrase", label="Phrase")
output = Textbox(label="Result")

iface = gr.Interface(fn=predictor, inputs=[input1, input2], outputs=[output], live=False,)
iface.launch()