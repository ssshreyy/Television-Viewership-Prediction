import os
import json
import pandas as pd
import LiveTweetSearch
import Sentiment_Analysis
import Tweet_Preprocessing
from flask_cors import CORS
from flask_restful import Api
from flask import Flask, request


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
api = Api(app)
CORS(app)

@app.route("/wordcloud", methods = ['POST','GET'])
def wordcloud():
    if request.method == 'POST':
        show = request.form.get('show')
        year = request.form.get('year')

        path = os.path.realpath('/assets/images/tweet-' + year + '-wordcloud.png')
        return path

    else:
        return 0

@app.route("/show", methods = ['POST','GET'])
def show():
    if request.method == 'GET':
        data = pd.read_csv('Tweet_data/tweet_data.csv', usecols=range(12), index_col=False, low_memory=False)
        return pd.DataFrame.to_json(data, orient='index')

@app.route("/preprocess", methods = ['POST','GET'])
def preprocess():
    if request.method == 'GET':
        Tweet_Preprocessing.main('Tweet_data/tweet_data.csv')
        data = pd.read_csv('Preprocessed_data/tweet_data_preprocessed.csv', usecols=range(13), index_col=False, low_memory=False)
        return pd.DataFrame.to_json(data, orient='index')

@app.route("/scatter", methods = ['POST','GET'])
def scatter():
    if request.method == 'GET':
        data = pd.read_csv('Prediction_data/predicted_file.csv', usecols=range(21), index_col=False, low_memory=False)
        return pd.DataFrame.to_json(data, orient='index')

@app.route("/line1", methods = ['POST','GET'])
def line1():
    if request.method == 'GET':
        res=[]
        yearList=['2009','2010','2011','2012','2013','2014','2015','2016','2017']
        for x in yearList:
            with open('Tweet_Data/tweet_'+x+'.csv') as f:
                res.append(sum(1 for line in f))
        return json.dumps(res)

@app.route("/line2", methods = ['POST','GET'])
def line2():
    if request.method == 'GET':
        res=[]
        data = pd.read_csv('Prediction_data/simpsons_episodes.csv', usecols=range(19), index_col=False, low_memory=False)
        data.dropna(inplace = True)
        res = {
            "ep": data['Air_Date'].tolist(),
            "imdb": data['IMDB_Rating'].tolist()
        }
        return json.dumps(res)


@app.route("/line3", methods = ['POST','GET'])
def line3():
    if request.method == 'GET':
        res=[]
        data = pd.read_csv('Prediction_data/simpsons_episodes.csv', usecols=range(19), index_col=False, low_memory=False)
        data.dropna(inplace = True)
        res = {
            "ep": data['Air_Date'].tolist(),
            "views": data['US_Viewers_In_Millions'].tolist()
        }
        return json.dumps(res)

@app.route("/bar", methods = ['POST','GET'])
def bar():
    if request.method == 'GET':
        data = pd.read_csv('Prediction_data/predicted_file.csv', usecols=range(21), index_col=False, low_memory=False)
        return pd.DataFrame.to_json(data, orient='index')


@app.route("/bar2", methods = ['POST','GET'])
def bar2():
    if request.method == 'GET':
        fileNames = ['2009','2010','2009','2010','2011','2012','2013','2014','2015','2016','2017']
        df = pd.DataFrame()
        pos=[]
        neg=[]
        for x in fileNames:
            data = pd.read_csv('Prediction_data/tweet_'+x+'_predict.csv', usecols=range(15), index_col=False, low_memory=False)
            temp = data['Sentiment_Score'].value_counts()
            # print(temp)
            pos.append(temp[4])
            neg.append(temp[0])
        df['Year'] = fileNames
        df['Pos'] = pos
        df['Neg'] = neg
        print(df)
        return pd.DataFrame.to_json(df, orient='index')

@app.route("/sentiment", methods = ['POST','GET'])
def sentiment():
    if request.method == 'GET':
        Sentiment_Analysis.main('./Preprocessed_data/preprocessed_training_data.csv')
        data = pd.read_csv('Prediction_data/tweet_data_predict.csv', usecols=range(15), index_col=False, low_memory=False)
        return(pd.DataFrame.to_json(data, orient='index'))

@app.route("/search", methods = ['POST','GET'])
def search():
    if request.method == 'POST':
        username = request.form.get('username')
        query = request.form.get('query')
        since = request.form.get('since')
        until = request.form.get('until')
        maxNo = request.form.get('maxNo')
        top = request.form.get('top')
        tweetSearchParameters=[]
        if username:
            tweetSearchParameters.append('--username')
            tweetSearchParameters.append(username)
        if query:
            tweetSearchParameters.append('--query')
            tweetSearchParameters.append(query)
        if since:
            tweetSearchParameters.append('--since')
            tweetSearchParameters.append(since)
        if until:
            tweetSearchParameters.append('--until')
            tweetSearchParameters.append(until)
        if maxNo:
            tweetSearchParameters.append('--maxtweets')
            tweetSearchParameters.append(maxNo)
        if top:
            tweetSearchParameters.append('--toptweets')

        LiveTweetSearch.main(tweetSearchParameters)

        data = pd.read_csv('Tweet_data/tweet_data.csv', usecols=range(12), index_col=False, low_memory=False)
        # return jsonify([{'tweetSearchParameters':data['Text'][1]}])
        return pd.DataFrame.to_json(data, orient='index')

    else:
        return 0


if __name__ == '__main__':
     app.run(port=5003)
