# -*- coding: utf-8 -*-
"""
@author: Laurence Berville 
"""
#%% Packages
import streamlit as st
import pandas as pd

from sklearn.pipeline import Pipeline # pour faire un pipeline
from sklearn.model_selection import train_test_split # pour diviser le dataset en training data set
from sklearn.feature_extraction.text import TfidfVectorizer # pour vectoriser les messages en matrice
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import string

import nltk# Preprocessing pour enlever ponctuation - Natural Language Processing 
nltk.download("punkt")
nltk.download('stopwords')
from nltk.corpus import stopwords # Preprocessing pour enlever les mots "courant"

#%% data
data= pd.read_csv('https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection', 
         sep='\t', 
    names=['Target','SMS'])
data=data.rename(columns={0:"Target",1:"SMS"}) # Renommer les colonnes.

#%% Preprocessing
# Les stops words
stop_words = list(stopwords.words('english'))
# enlever les stopwords et les ponctiations du "texte des sms"
def text_process(text): # pour enlever les mots les plus courants et la ponctuation des messages
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)
df= pd.DataFrame(data['SMS'].apply(text_process))

# Sauvegarder les labels et les mettre sous forme dataframe avec les sms "propres".
label = pd.DataFrame(data['Target']) # Colonne avec que les Targets
df= pd.concat([label, df],axis=1) 
df = df.drop_duplicates()# enlever les doublons


#%% label encoder

lb_encod = LabelEncoder()
y = lb_encod.fit_transform(df['Target']) # transformer les labels en 0 / 1.

#%% déclarer les features

X=pd.DataFrame(df['SMS'])

# target preprocessing
tfidf_vectorizer  = TfidfVectorizer(use_idf=False, 
                                        lowercase=True, 
                                        strip_accents='ascii')
                                        #,stop_words=stop_words) # enlever les stopwords
                                        
#%% Train set split :
X_train, X_test, y_train, y_test = train_test_split(data['SMS'], 
                                                    data['Target'],
                                                    test_size=0.20,# 20% test size et 80% train
                                                    random_state=42,
                                                    stratify=data['Target']) #
#%% Pipeline : 
svc = SVC(kernel='sigmoid', gamma=1.0)

SVC_vectorizer = Pipeline([
     ('vectorizer', tfidf_vectorizer),
     ('classifier', svc)
 ])

#%% Fit le modèle : 
SVC_vectorizer.fit(X_train, y_train)

#%% Prédiction : 
y_pred_SVC = SVC_vectorizer.predict(X_test)

#new_sms =input()
#prediction = SVC_vectorizer.predict([new_sms])




# # Affichage ---------------------------------
# st.write ("Application  - Détermination des spams")

# st.text_input('Veuillez entrer votre sms suspect :')

# if prediction[0] == "spam":
#     print("This sms is spam.")
# else:
#     print("This sms is not spam.")


