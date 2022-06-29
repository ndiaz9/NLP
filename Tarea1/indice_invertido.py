# %% leer datos 
from re import M
import pandas as pd
import numpy as np

import xml.etree.ElementTree as ET
import os
import nltk
from gensim.parsing.porter import PorterStemmer 

# get all documents from the directory
def get_documents(path):
    documents = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".naf"):
            tree = ET.parse(path+filename)
            texto = tree.find('raw').text
            documents.append(texto)
    return documents

datos = get_documents('datos/docs-raw-texts/')
documentos = pd.DataFrame(datos,columns=['Documento'])
# %%
documentos.head()
# %%

from nltk.corpus import stopwords


def remove_stopwords(data):
    filtered_words = [word.lower() for word in data.split(' ') if word.lower() not in stopwords.words('english')]    
    return filtered_words

def preprocesar(documentos):
# 2. Preprocess the data
#remover espacios dobles y triples
    import re
    documentos = re.sub('\n', ' ',documentos)
    documentos = re.sub('[^a-zA-Z]|[0-9]', ' ',documentos)
    documentos = re.sub('\s+', ' ',documentos)
    p=PorterStemmer()
    documentos = p.stem_sentence(documentos)

    filtrada= remove_stopwords(documentos)
    
    return filtrada
# %% preparar datos
documentos['filtrada']=documentos['Documento'].apply(preprocesar)
# %% preparar datos
doc_proc= documentos
#%% remover palabras duplicadas y ordenar
doc_proc.filtrada = doc_proc.filtrada.apply(np.unique)
doc_proc.head()

# %% 
indice_invertido = {}

for i in range(len(doc_proc)):
    for j in range(len(doc_proc.iloc[i]['filtrada'])):
        if doc_proc.iloc[i]['filtrada'][j] not in indice_invertido:
            indice_invertido[doc_proc.iloc[i]['filtrada'][j]] = []
        indice_invertido[doc_proc.iloc[i]['filtrada'][j]].append(i+1)
#%% save dict to file
import json
a_file = open("salida/indice_invertido.json", "w")
json.dump(indice_invertido, a_file)
a_file.close()
#%% read dict from file
a_file = open("salida/indice_invertido.json", "r")
indice_invertido = json.loads(a_file.read())
#%%
datos_querry = get_documents('datos/queries-raw-texts/')
queries = pd.DataFrame(datos_querry,columns=['Query'])

queries
# %%
queries['filtrada'] = queries.Query.apply(preprocesar)
quer_proc = queries
# %%
quer_proc.filtrada = quer_proc.filtrada.apply(np.unique)
quer_proc.head()
# %%

def BSII_AND(lista_busqueda):
    if len(lista_busqueda) == 0:
        return ''
    if len(lista_busqueda) == 1:
        listaA = indice_invertido[lista_busqueda[0]]
        return ','.join([f'd{x:03}' for x in listaA])
    else:
        listaA = indice_invertido[lista_busqueda[0]].copy()
        for i in range(1,len(lista_busqueda)):
            if lista_busqueda[i] in indice_invertido:
                listaB = indice_invertido[lista_busqueda[i]]
                listaC = []
                for elem in listaA:
                    if elem in listaB:
                        listaC.append(elem)
                listaA = listaC.copy()
            else:
                return ''
        
        return ','.join([f'd{x:03}' for x in listaA])
# %%
q=quer_proc.filtrada.apply(BSII_AND)
# %%
f = open("salida/BSII-AND-queries_results.txt", "w")
for i in range(len(q)):
    f.write(f'q{i+1:02} {q[i]}\n')
f.close()
# %%

def BSII_OR(lista_busqueda):
    if len(lista_busqueda) == 0:
        return ''
    if len(lista_busqueda) == 1:
        listaA = indice_invertido[lista_busqueda[0]]
        return ','.join([f'd{x:03}' for x in listaA])
    else:
        listaA = indice_invertido[lista_busqueda[0]].copy()
        for i in range(1,len(lista_busqueda)):
            if lista_busqueda[i] in indice_invertido:
                listaB = indice_invertido[lista_busqueda[i]]
                listaC =  listaA.copy()+listaB.copy()
                listaA = listaC.copy()
            else:
                return ''
        listaA = np.unique(listaA)        
        return ','.join([f'd{x:03}' for x in listaA])

# %%
q_or=quer_proc.filtrada.apply(BSII_OR)

# %%
f = open("salida/BSII-OR-queries_results.txt", "w")
for i in range(len(q_or)):
    f.write(f'q{i+1:02} {q_or[i]}\n')
f.close()
# %%
