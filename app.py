from flask import Flask, request, jsonify
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW, get_scheduler
from sklearn import preprocessing
import urllib.parse
import requests
from config import OuterService



device = torch.device("cpu")
model_name = "sberbank-ai/sbert_large_nlu_ru"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=660).to(device)
Label_encoder = preprocessing.LabelEncoder()
Label_encoder.classes_ = np.load('codetnved/src/cl_classes2306.npy', allow_pickle=True)
model.load_state_dict(torch.load("pytorch_model.bin", map_location=device))

model_10d = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2648).to(device)
Label_encoder_10d = preprocessing.LabelEncoder()
Label_encoder_10d.classes_ = np.load('codetnved/src/cl_classes1307.npy', allow_pickle=True)
model_10d.load_state_dict(torch.load("pytorch_model10d.bin", map_location=device))

app = Flask(__name__)
dscr = pd.read_csv("codetnved/data/desc.csv", sep=';', names=['id', 'label'])

def predict_prob(text, qtty=2):
    model.to(torch.device('cpu'))
    inputs = tokenizer(text, truncation = True, max_length=100, padding='max_length', return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    result = dict()
    p = torch.nn.functional.softmax(logits, dim=1)
    for i in range(qtty):
        a = p.argmax().item()
        result[Label_encoder.inverse_transform([a])[0]] = p[0][a].item()
        p[0][a] = 0
    return result
def predict_prob_with_descr(text, qtty=5):
    probs = predict_prob(text, qtty=qtty)
    #result = np.array()
    result = list()
    for each in probs:
        result.append([each, dscr[dscr['id']==each].iloc[0]['label'], probs[each]])
    return result


def getCodeOuterService(text):
    text = urllib.parse.quote(text)
    url = f'{OutsideService}={text}'
    payload = {}
    headers = {
    'Cookie': 'PHPSESSID=7NoMuGpXzqaSNjIakhZNcAtrik8lSKDm'
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    return response.json()


@app.route('/GetGroupCode/', methods=['POST'])
def GetGroupCode():

    data = request.get_json()
    if 'text' not in data:
        return jsonify({'code': 'Не был передан необходимый параметр'})
    if 'qty' not in data:
        codes = predict_prob_with_descr(data['text'])
    else:
        codes = predict_prob_with_descr(data['text'], data['qty'])
    CodeOuterService = getCodeOuterService(data['text'])
    return jsonify({'Our':codes , 'OuterService':CodeOuterService})

@app.route('/GetCode/', methods=['POST'])
def GetCode():

    data = request.get_json()
    if 'text' not in data:
        return jsonify({'code': 'Не был передан необходимый параметр'})
    return jsonify({'code': 'Не реализовано'})


if __name__ == '__main__':
   app.run(debug=False)
    
