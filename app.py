from flask_cors import CORS
from flask import Flask, request
import json
import datetime
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from google.cloud import firestore
import os
from nlp import FooModel, FooNLP, W2VModel

#
# word2vec and tfidf serialized models
#

app = Flask(__name__)
CORS(app)

nlp = FooNLP()
models = {}
dir = '/azmodels'

for f in ['tfidf.nb', 'w2vcbow.lr']:
    path = f'{dir}/{f}.fulltwitter.foonlp.ser'
    if (os.path.exists(path)):
        print(f'----- loading model {path}')
        models[f] = nlp.load(path)
    else:
        print(f'----- creating/saving model {path}')
        nlp.load_train_twitter(500000, f'{dir}/SentimentAnalysisDataset.csv')
        nlp.save(path, nlp)   
        models[f] = nlp
print('loaded models', models)


@app.route('/', methods=['GET'])
def base():
    return '<div>Welcome to the Flask NLP -- paths available:  /nlp/sa/ <P> Models loaded:<P/><P/><PRE> ' +str(models)+ '</PRE></div>'


@app.route('/nlp/history', methods=['GET'])
def sa_history(max=50):
    print('sa_history')
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gkey.json"
    db = firestore.Client()
    print('db')
    resp = []

    users_ref = db.collection(u'queries').order_by(u'date', direction=firestore.Query.DESCENDING).limit(max)
    print('ref')
    for doc in users_ref.stream():
        resp.append(doc.to_dict())
        print(u'{} => {}'.format(doc.id, doc.to_dict()))
    return json.dumps(resp)


@app.route('/nlp/save', methods=['POST'])
def sa_save():
    print('sa_save')
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gkey.json"
    db = firestore.Client()

    j_data = request.get_json()
    print(j_data)

    doc_ref = db.collection(u'queries')
    doc_ref.add(j_data)

    return str(j_data)


@app.route('/nlp/sa/<model>', methods=['GET'])
def sa_predict(model='all'):
    sentence = request.args.get('data')
    print(sentence)

    resp = {}
    resp['input'] = sentence
    resp['results'] = []

    if (model == 'all'):
        resp['results'].append(vader(sentence))
        resp['results'].append(textblob(sentence))
        resp['results'].append(azure_sentiment(sentence))
        resp['results'].append(gcp_sentiment(sentence))
        resp['results'].append(custom_nlp1(sentence))
        resp['results'].append(custom_nlp2(sentence))

    elif (model == 'azure'):
        resp['results'] = azure_sentiment(sentence)
    elif (model == 'vader'):
        resp['results'] = vader(sentence)
    elif (model == 'textblob'):
        resp['results'] = textblob(sentence)
    elif (model == 'google'):
        resp['results'] = gcp_sentiment(sentence)
    elif (model == 'w2vcbow.lr'):
        resp['results'] = custom_nlp1(sentence)
    elif (model == 'tfidf.nb'):
        resp['results'] = custom_nlp2(sentence)
    else:
        # flag error 
        return 'No Model exists for '+model

    # db save to firestore (should re-use save rest service?)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gkey.json"
    db = firestore.Client()
    j_data = {"from": "anonymous", "text": sentence, "scores": resp['results'],"date": datetime.datetime.now().isoformat("T")}

    doc_ref = db.collection(u'queries')
    doc_ref.add(j_data)

    return json.dumps(resp)


def textblob(sentence):
    resp = {}
    resp['model'] = 'TextBlob'
    resp['extra'] = 'models returns -1 to +1'
    resp['url'] = 'https://textblob.readthedocs.io/en/dev/'
    # create TextBlob object of passed tweet text
    analysis = TextBlob(sentence)
    resp['rScore'] = analysis.sentiment.polarity
    resp['nScore'] = analysis.sentiment.polarity
    # set sentiment
    return resp


def vader(sentence):
    resp = {}
    resp['model'] = 'Vader'
    resp['extra'] = 'model returns -1 to +1'
    resp['url'] = 'https://pypi.org/project/vaderSentiment/'
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)['compound'] 
    resp['rScore'] = score
    resp['nScore'] = score
    return resp


# azure and google inspired from:  https://www.pingshiuanchua.com/blog/post/simple-sentiment-analysis-python?utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com
##
def gcp_sentiment(text):
    resp = {}
    resp['model'] = 'Google NLP'
    resp['extra'] = 'model returns -1 to +1'
    resp['url'] = 'https://cloud.google.com/natural-language/'

    gcp_url = "https://language.googleapis.com/v1/documents:analyzeSentiment?key=AIzaSyBN-SLv7YPAMARDo2eQl7Y_yyy84xpWcHU"

    document = {'document': {'type': 'PLAIN_TEXT', 'content': text}, 'encodingType':'UTF8'}
    response = requests.post(gcp_url, json=document)
    sentiments = response.json()
    score = sentiments['documentSentiment']['score']

    resp['rScore'] = score
    resp['nScore'] = score 
    return resp


# azure service calls
##
def azure_sentiment(text):
    resp = {}
    resp['model'] = 'Azure NLP'
    resp['extra'] = 'model returns 0 to 1'
    resp['url'] = 'https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/'

    documents = {'documents': [
        {'id': '1', 'text': text}
    ]}

    azure_key = 'd6c00eb74e58455187125aa6a97fd976'  # Update here
    azure_endpoint = 'https://textsentimentanalyzer.cognitiveservices.azure.com/text/analytics/v2.1/'
    sentiment_azure = azure_endpoint + '/sentiment'

    headers = {"Ocp-Apim-Subscription-Key": azure_key}
    response = requests.post(sentiment_azure, headers=headers, json=documents)
    score = response.json()['documents'][0]['score']

    resp['rScore'] = score
    resp['nScore'] = 2 * (score - 0.5) 

    return resp


def custom_nlp1(text):
    n = models['w2vcbow.lr']
    label, prob = n.predict([text])
    print(label, prob)

    resp = {}
    resp['model'] = 'Foo W2V LR'
    resp['extra'] = 'model returns 0 to 1'
    resp['url'] = 'http://foostack.ai/'
    resp['rScore'] = prob[0][1]
    resp['nScore'] = 2 * (resp['rScore'] - 0.5) 

    return resp


def custom_nlp2(text):
    n = models['tfidf.nb']
    label, prob = n.predict([text])
    print(label, prob)

    resp = {}
    resp['model'] = 'Foo TFIDF NB'
    resp['extra'] = 'model returns 0 to 1'
    resp['url'] = 'http://foostack.ai/'
    resp['rScore'] = prob[0][1]
    resp['nScore'] = 2 * (resp['rScore'] - 0.5) 

    return resp


if __name__ == '__main__':
    app.run(debug=True)
