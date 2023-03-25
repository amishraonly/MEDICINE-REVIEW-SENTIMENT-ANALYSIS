import pickle

import pandas as pd
from flask import Flask, render_template, request

import re

import nltk
nltk.download('stopwords')
# For tokenization
nltk.download('punkt')
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
from tqdm import tqdm
def preprocess_text(text_data):
    preprocessed_text = []
    # tqdm is for printing the status bar
    for sentance in tqdm(text_data.split(' ')):
        #print("sentence",sentance)
        sent = decontracted(sentance)
        sent = sent.replace(' ', '')
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\n', ' ')
        sent = sent.replace('\\"', ' ')
        sent = sent.replace('-',' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        print(sent)
        sent = ''.join(e for e in sent.split() if e.lower() not in stopwords)
        print("SFD",sent)
        preprocessed_text.append(sent.lower().strip())
    return preprocessed_text






app = Flask(__name__)
@app.route("/", methods=["GET"])
def root():
    return render_template("index.html")
@app.route("/predict", methods=["GET"])
def predict():
    print("yes")
    drugname = request.args.get('drugname')
    condition = request.args.get('condition')
    review = request.args.get('review')
    d = {'r_data':review}
    df = pd.DataFrame(d, index=[0])
    d['r_data'] = preprocess_text(d['r_data'])
    print(d['r_data'])
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    test_review = tfidf_vectorizer.transform(d['r_data'])

    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    a = loaded_model.predict(test_review)
    y = str(a[0])
    if y == 0:
        return "Review is bad"
    else:
        return "Review is good"



app.run(port=8081, debug=True, host="0.0.0.0")
