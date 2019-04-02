import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from scipy.sparse import hstack
from nltk.sentiment import vader
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


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


def calculate_vader(tweet):

    if type(float) == type(tweet):
        return 0
    sia = vader.SentimentIntensityAnalyzer()
    return sia.polarity_scores(tweet)['compound']


def main(fileName):

    print('Sentiment Analysis Model Training Started')
    inputFileName = './Preprocessed_data/tweet_data_preprocessed.csv'
    outputFileName = './Prediction_data/tweet_data_predict.csv'
    algorithm = 'Logistic_Regression'

    train_dataset = pd.read_csv(fileName, usecols = range(7), encoding = 'Latin-1', index_col = False, low_memory = False)
    train_dataset.Tidy_Tweet = train_dataset.Tidy_Tweet.fillna(value="")
    print('Preprocessed Sentiment Training File read')

    x = np.array(train_dataset.Tidy_Tweet)
    y = np.array(train_dataset.sentiment)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    data_train = x_train
    label_train = y_train
    data_test = x_test
    label_test = y_test
    print('Data Sliced In Training And Testing Sets')

    tfv = TfidfVectorizer(sublinear_tf = True , stop_words = "english")
    Tfidf_features_train = tfv.fit_transform(data_train)
    Tfidf_features_test = tfv.transform(data_test)
    print("TF-IDF Features Extracted")

    bow_vectorizer = CountVectorizer(max_df = 0.90, min_df = 2, max_features = 1000, stop_words = 'english')
    bow_features_train = bow_vectorizer.fit_transform(data_train)
    bow_features_test = bow_vectorizer.transform(data_test)
    print("BOW Features Extracted")

    features_final_train = hstack((Tfidf_features_train, bow_features_train))
    features_final_test = hstack((Tfidf_features_test, bow_features_test))
    print("Training And Testing Sparse Matrix Created")

    # print("Model Training Started")
    # model = train_classifier(features_final_train, features_final_test, label_train, label_test, algorithm)
    # print("Model Training Complete")

    fileName = './Sentiment_models/' + algorithm + '.pickle'
    pickle_in = open(fileName, 'rb')
    model = pickle.load(pickle_in)
    print("%s Model Loaded" % algorithm)

    prediction_dataset = pd.read_csv(inputFileName, usecols = range(13), encoding = 'Latin-1', index_col = False, low_memory = False)
    prediction_dataset.Tidy_Tweet = prediction_dataset.Tidy_Tweet.fillna(value = "")
    x_prediction = np.array(prediction_dataset.Tidy_Tweet)
    print("Input Tweet File Read")

    features_x_prediction1 = tfv.transform(x_prediction)
    features_x_prediction2 = bow_vectorizer.transform(x_prediction)
    features_x_prediction = hstack((features_x_prediction1, features_x_prediction2))
    print("Sparse Matrix Merged")

    prediction_dataset['Sentiment_Score'] = model.predict(features_x_prediction)
    print("Sentimental Analysis Using %s Completed" % algorithm)

    prediction_dataset['Vader_Score'] = prediction_dataset['Tidy_Tweet'].apply(calculate_vader)
    print("Sentimental Analysis Using Vader Completed")

    prediction_dataset.to_csv(outputFileName, index = False)
    print("Sentimental Analysis Of Tweets Complete. Output File Generated %s" % outputFileName)
    return outputFileName


if __name__ == '__main__':
    main('./Preprocessed_data/preprocessed_training_data.csv')
