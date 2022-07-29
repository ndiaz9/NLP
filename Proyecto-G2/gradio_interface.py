import pickle, gensim, re
import gradio as gr
import pandas as pd
import numpy as np
from gradio.components import Dropdown, Textbox
from sklearn.preprocessing import Normalizer
from keras.models import load_model
from keras.preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag, download

download('wordnet')
download('sentiwordnet')
download('omw-1.4')
download('averaged_perceptron_tagger')
glove_embedding = KeyedVectors.load_word2vec_format('./datos/glove.6B.100d.txt.word2vec', binary=False)
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


def expand_contractions(document: str) -> str:
    """
    Replace all abbreviations with their corresponding expansion
    """
    document = re.sub(r"'cause", "because", document)
    document = re.sub(r"o'clock", "of the clock", document)
    document = re.sub(r"won\'t", "will not", document)
    document = re.sub(r"can\'t", "can not", document)
    document = re.sub(r"n\'t", " not", document)
    document = re.sub(r"\'re", " are", document)
    document = re.sub(r"\'s", " is", document)
    document = re.sub(r"\'d", " would", document)
    document = re.sub(r"\'ll", " will", document)
    document = re.sub(r"\'t", " not", document)
    document = re.sub(r"\'ve", " have", document)
    document = re.sub(r"\'m", " am", document)
    return document


def replace_numbers(document: str) -> str:
    """
    Replace number appearances with 'number'
    """
    # Case 1: Combination of numbers and letters (Eg. 2nd -> NUM)
    document = re.sub('[a-zA-Z]+[0-9]+[a-zA-Z]+', 'number', document)
    document = re.sub('[0-9]+[a-zA-Z]+|[a-zA-Z]+[0-9]+', 'number', document)
    # Case 2: Decimal numbers (Eg. 2.1 -> NUM)
    document = re.sub('[0-9]+\.+[0-9]+', 'number', document)
    # Case 3: Numbers between spaces (Eg. 220 888 -> NUM)
    document = re.sub('([0-9]+\s)*[0-9]+', 'number', document)
    # Case 4: One or more of the previous cases (Eg. NUM NUM -> NUM)
    document = re.sub('((NUM)+\s)*(NUM)+', 'number', document)
    return document


def preprocessing(document: str) -> list:
    """
    iterate over all words in document identifing the word and frecuency
    remove all the problematic characters over the word
    and return a dictionary with the word as the key and the frecuency as the value
    """
    document = document.lower()
    document = expand_contractions(document)
    document = replace_numbers(document)
    document = re.sub('[^A-Za-z0-9]+', ' ', document)
    document = document.split()
    return document


def sentence_to_embedding(sentence: str, embedding: bin) -> np.array:
    """
    Returns the element-wise mean of the embeddings that represent each word in a sentence
    """
    words = preprocessing(sentence)
    vector = np.zeros(embedding.layer1_size)
    counter = 0
    for word in words:
        try:
            vector += embedding.wv[word]
            counter += 1
        except:
            pass
    if counter > 0:
        vector = vector / counter
    return vector


def glove_sentence_to_embedding(sentence: str, embedding: bin) -> np.array:
    """
    Returns the element-wise mean of the embeddings that represent each word in a sentence
    """
    words = preprocessing(sentence)
    vector = np.zeros(embedding.vector_size)
    counter = 0
    for word in words:
        try:
            vector += embedding.get_vector(word)
            counter += 1
        except Exception as e:
            pass
    if counter > 0:
        vector = vector / counter
    return vector


def predictor(text):
    # "Naive Bayes"
    count_vector = pickle.load(open('salida/naive_bayes.vector', 'rb'))
    naive_bayes = pickle.load(open('salida/naive_bayes.model', 'rb'))
    data = count_vector.transform([text])
    norm = Normalizer().fit_transform(data)
    response1 = 'Hate Speech' if naive_bayes.predict(norm)[0] == 1 else 'Not Hate Speech'
    # "Logistic Regression"
    count_vector = pickle.load(open('salida/logistic_regression.vector', 'rb'))
    logistic_regression = pickle.load(open('salida/logistic_regression.model', 'rb'))
    data = count_vector.transform([text])
    norm = Normalizer().fit_transform(data)
    response2 = 'Hate Speech' if logistic_regression.predict(norm)[0] == 1 else 'Not Hate Speech'
    # "Lexicon"
    naive_bayes = pickle.load(open('salida/swn_lexicon_naive_bayes.model', 'rb'))
    data = lexicon_sentiwordnet(text) if text else [0,0,0,0,0,0,0,0,0,0,0,0,0]
    response3 = 'Hate Speech' if naive_bayes.predict([data])[0] == 1 and text != '' else 'Not Hate Speech'
    # "Random Forest"
    count_vector = pickle.load(open('salida/random_forest.vector', 'rb'))
    random_forest = pickle.load(open('salida/random_forest.model', 'rb'))
    data = count_vector.transform([text])
    norm = Normalizer().fit_transform(data)
    response4 = 'Hate Speech' if random_forest.predict(norm)[0] == 1 else 'Not Hate Speech'
    # "Trained Embedding"
    embedding = gensim.models.Word2Vec.load('salida/embedding.model')
    model = load_model('salida/embedding_model')
    data = np.asarray([sentence_to_embedding(text, embedding).tolist()]).astype('float32')
    pred = model.predict(data, verbose=1)
    response5 = 'Hate Speech' if pred > 0.5 else 'Not Hate Speech'
    # "Glove Embedding"
    glove_model = load_model('salida/embedding_glove')
    emddng = np.asarray([glove_sentence_to_embedding(text, glove_embedding).tolist()]).astype('float32')
    pred = glove_model.predict(emddng, verbose=1)
    response6 = 'Hate Speech' if pred > 0.5 else 'Not Hate Speech'
    # "RNN"
    word_token = pickle.load(open('salida/word_token.vector', 'rb'))
    rnn_model = load_model('salida/RNN_model')
    serie_text = pd.Series([preprocessing(text)])
    sequences = word_token.texts_to_sequences(serie_text)
    sequences_matrix = pad_sequences(sequences,maxlen=150)
    pred = rnn_model.predict(sequences_matrix)
    response7 = 'Hate Speech' if pred[0][0] > 0.25 else 'Not Hate Speech'
    return [response1, response2, response3, response4, response5, response6, response7]

# input1 = Dropdown(["Naive Bayes", "Logistic Regression", "Lexicon", "Random Forest", "Trained Embedding", "Glove Embedding", "RNN"], label="Model")
input = Textbox(placeholder="Enter Phrase", label="Phrase")
output1 = Textbox(label="Naive Bayes")
output2 = Textbox(label="Logistic Regression")
output3 = Textbox(label="Lexicon")
output4 = Textbox(label="Random Forest")
output5 = Textbox(label="Trained Embedding")
output6 = Textbox(label="Glove Embedding")
output7 = Textbox(label="RNN")

iface = gr.Interface(
    fn=predictor, 
    inputs=[input], 
    outputs=[output1, output2, output3, output4, output5, output6, output7], 
    live=False, 
    title="Hate Speech Detector",
    description="Hate Speech Detector using Machine Learning & Natural Language Processing techniques"
)
iface.launch()