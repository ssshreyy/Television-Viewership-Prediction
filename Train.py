import pickle
import pandas as pd
import bisect, datetime
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_classifier(features_train, features_test, label_train, label_test, classifier):
    if classifier == "Logistic_Regression":
        model = LogisticRegression(C=1.)
    elif classifier == "Naive_Bayes":
        model = MultinomialNB()
    elif classifier == "SVM":
        model = SVC()
    elif classifier == "Random_Forest":
        model = RandomForestClassifier(n_estimators=400, random_state=11)
    else:
        print("Incorrect Selection Of Classifier")

    model.fit(features_train, label_train)
    print("Model Fitting Done")

    fileName = './Sentiment_models/' + classifier + '.pickle'
    with open(fileName, 'wb') as file:
        pickle.dump(model, file)
    print("Pickle File Created %s" % fileName)

    accuracy = model.score(features_test, label_test)
    print("Accuracy Is:", accuracy)

    # Make prediction on the test data
    probability_to_be_positive = model.predict_proba(features_test)[:,1]

    # Check AUC(Area Under the Roc Curve) to see how well the score discriminates between negative and positive
    print("AUC (Train Data):", roc_auc_score(label_test, probability_to_be_positive))

    # Print top 10 scores as a sanity check
    print("Top 10 Scores: ", probability_to_be_positive[:10])

    return model


def computeAverage(tweet_data, start, end):
    if start == end:
        return 0
    total = end-start+1
    score = 0
    count = 1
    for i in range(start, end+1):
        score += count * float(tweet_data['Vader_Score'][i])
        count += 1

    return score/total


def date_change(str_date):
    if str_date:
        return datetime.datetime.strptime(str_date, '%d-%m-%Y').strftime('%Y-%m-%d')


def viewers_change(str_views):
    if str_views == 'NaN':
        return '0'
    return str(int(float(str_views) * 1000000))


# def date_change(str_date):
#     return datetime.datetime.strptime(str_date, '%B %d, %Y')


# def viewers_change(str_views):
#     return str(int(float(str_views.strip().split('[')[0]) * 1000000))


def main(prediction_file, simpsons_file):

    algorithm = "Logistic_Regression"
    print('Viewership Prediction Started')
    viewer_data = pd.read_csv(simpsons_file, usecols=range(13), index_col=False, low_memory = False)
    print('Episode Data File Read Successful')

    tweet_data = pd.read_csv(prediction_file, usecols=range(15), index_col=False, low_memory = False)
    print('Tweet Data File Read Successful')

    viewer_data['Air_Date'] = list(map(date_change, viewer_data['Air_Date']))
    tweet_data['Date'] = list(map(date_change, tweet_data['Date']))
    print('Date Columns Altered')

    viewer_data['US_Viewers_In_Millions'] = list(map(viewers_change, viewer_data['US_Viewers_In_Millions']))
    print('Viewer Column Altered')

    first_date = bisect.bisect_left(viewer_data['Air_Date'], '2009-01-01')
    last_date = bisect.bisect_left(viewer_data['Air_Date'], '2015-01-01')
    y_train = list(map(int, viewer_data['US_Viewers_In_Millions'][first_date+1:last_date]))

    x_train = list()
    count = 1
    print('Extracting Training Features')
    for i in range(first_date, last_date - 1):
        temp1 = str(viewer_data['Air_Date'][i])
        temp2 = str(viewer_data['Air_Date'][i + 1])
        temp3 = []
        start = bisect.bisect_left(tweet_data['Date'], temp1)
        end = bisect.bisect_left(tweet_data['Date'], temp2)
        temp3.append(computeAverage(tweet_data, start, end))

        print(count, temp2, viewer_data['Title'][i + 1], temp3)

        count += 1
        x_train.append(temp3)

    print('4')

    first_date = bisect.bisect_left(viewer_data['Air_Date'], '2015-01-01')
    last_date = bisect.bisect_left(viewer_data['Air_Date'], '2016-01-01')

    y_test = list(map(int, viewer_data['US_Viewers_In_Millions'][first_date + 1:last_date]))

    x_test = list()
    print('6')
    count = 1
    for i in range(first_date, last_date - 1):
        temp1 = str(viewer_data['Air_Date'][i])
        temp2 = str(viewer_data['Air_Date'][i + 1])
        temp3 = []
        start = bisect.bisect_left(tweet_data['Date'], temp1)
        end = bisect.bisect_left(tweet_data['Date'], temp2)
        temp3.append(computeAverage(tweet_data, start, end))

        print(count, temp2, viewer_data['Title'][i + 1], temp3)

        count += 1
        x_test.append(temp3)

    print('7')

    # model = train_classifier(x_train, y_train, x_test, y_test, algorithm)

    lin = linear_model.LinearRegression()
    lin.fit(x_train,x_test)
    acc = lin.score(x_train,x_test)
    print("Linear ",acc)

    lin = linear_model.LogisticRegression()
    lin.fit( x_train , x_test )
    acc = lin.score( x_train , x_test )
    print("Logistic ",acc)

if __name__ == "__main__":
    main('./Prediction_data/tweet_predict.csv', './Prediction_data/simpsons_episodes.csv')
