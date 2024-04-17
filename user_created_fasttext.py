from gensim.models.phrases import Phrases, Phraser
from gensim.models import FastText
import pandas as pd


df = pd.read_csv('medical_dataset.csv')
print(df.head())

sent = [row.split() for row in df['Text']]
phrases = Phrases(sent, min_count = 30, progress_per = 10000)
sentences = phrases[sent]


#Initializing the model
model = FastText(vector_size = 100, window = 5, min_count = 5, workers = 4, min_n = 1, max_n = 4)
#Building Vocabulary

model.build_vocab(sentences)
print(len(model.wv.index_to_key))

#Training the model
model.train(sentences, total_examples = len(sentences), epochs=100) 


# Saving the model

import joblib
path = 'FastText.joblib'
joblib.dump(model, path)

vocabulary = model.wv.index_to_key

for_py = model.wv.most_similar("python", topn = 5)

print(f"Most similar to python: {for_py} \n")

for_eob = model.wv.most_similar("epidemic out-break", topn = 10)
print(f"Similar to epidemic out-break: {for_eob} \n")