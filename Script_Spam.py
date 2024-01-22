# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:33:04 2024

@author: Mpadmin
"""
#%% Packages
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle
import pandas as pd
import numpy as np
import nltk # enlever ponctuation
import string
from nltk.corpus import stopwords


#%% Ouverture du projet - Mise en forme des données.

data = pd.read_csv('https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection', 
         sep='\t', header=None)

data=data.rename(columns={0:"Target",1:"SMS"})

#%% Vérification des données. 

Ratio=data['Target'].value_counts()

#_______________________________________________________________
SpamData = data[data['Target'] == "spam"] #sous tableau
HamData = data[data['Target'] == "ham"]

SpamData_counts= SpamData['SMS'].value_counts()# il y a des duplicats
HamData_counts= HamData['SMS'].value_counts()# il y a des duplicats

SpamData["Number of Words"] = SpamData['SMS'].apply(lambda n: len(n.split()))
mean_Spam = SpamData["Number of Words"].mean()
HamData["Number of Words"] = HamData["SMS"].apply(lambda n: len(n.split()))
mean_Ham = HamData["Number of Words"].mean()


#_______________________________________________________________
data["Number of Words"] = data["SMS"].apply(lambda n: len(n.split()))

sns.set_theme(style="ticks")

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(7, 6))

# Plot the orbital period with horizontal boxes
sns.boxplot(
    data, x="Number of Words", y="Target", hue="Target",
    whis=[0, 100], width=.6, palette="vlag")

# Add in points to show each observation
sns.stripplot(data, x="Number of Words", y="Target", size=2, color=".2")

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)

#%%
# Enlever la ponctuation.
nltk.download('stopwords')
stop_words = list(stopwords.words('english'))

#enlever la ponctuation :
def text_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

data_clean= data['SMS'].apply(text_process)

text = pd.DataFrame(data['SMS'])# colonne avec que les SMS
text1 = pd.DataFrame(data['SMS-clean'])# colonne avec que les SMS
label = pd.DataFrame(data['Target']) # Colonne avec que les Target

#%% Dénombrer le nombre de mot dans la colonne SMS

from collections import Counter

total_counts = Counter()
for i in range(len(text1)):
    for word in text1.values[i][0].split(" "):
        total_counts[word] += 1

print("Dans le data set, il y a : ", len(total_counts))


# total_counts = Counter()
# for i in range(len(text)):
#     for word in text.values[i][0].split(" "):
#         total_counts[word] += 1

# print("Dans le data set, il y a : ", len(total_counts))

#%% 

vocabulaireCourant = sorted(total_counts, key=total_counts.get, reverse=True)
print(vocabulaireCourant[:60])



