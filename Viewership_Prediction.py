import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def train_classifier(features_train, features_test, label_train, label_test, classifier):
    if classifier == "Logistic_Regression":
        model = LogisticRegression()
    elif classifier == "Naive_Bayes":
        model = MultinomialNB()
    elif classifier == "SVM":
        model = SVC()
    elif classifier == "Linear":
        model = LinearRegression()
    elif classifier == "Polynomial":
        poly = PolynomialFeatures(degree = 1)
        features_train = poly.fit_transform(features_train)
        features_test = poly.fit_transform(features_test)
        model = LinearRegression()
    elif classifier == "Random_Forest":
        model = RandomForestClassifier(n_estimators=400, random_state=11)
    elif classifier == "Kmeans":
        knn = neighbors.KNeighborsRegressor()
        params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
        model = GridSearchCV(knn, params, cv=5)
    else:
        print("Incorrect Selection Of Classifier")

    print("Model Selection Complete")
    model.fit(features_train, label_train)
    print("Model Fitting Done")

    fileName = './Prediction_models/' + classifier + '.pickle'
    with open(fileName, 'wb') as file:
        pickle.dump(model, file)
    print("Pickle File Created %s" % fileName)

    accuracy = model.score(features_test, label_test)
    print("Accuracy Is:", accuracy)

    return model


def main(simpsons_file):

    print('Viewership Prediction Started')
    viewer_data = pd.read_csv(simpsons_file, dtype={'Unique_Users': float, 'US_Viewers_In_Millions': float}, usecols = range(19), index_col = False, low_memory = False)
    viewer_data.dropna( inplace = True )
    print('Episode Data File Read Successful')

    x = viewer_data.loc[:, ['Views', 'IMDB_Rating', 'IMDB_Votes', 'Retweets', 'Favorites', 'Vader_Score', 'Sentiment_Score', 'Tweets_Per_Day', 'Unique_Users']]
    y = viewer_data.loc[:, ['US_Viewers_In_Millions']]
    # print(y)
    scaler = MinMaxScaler( feature_range = (0, 1))
    x = scaler.fit_transform(x)
    # y = scaler.fit_transform(y)
    print('Data Rescaling Complete')

    x = preprocessing.scale(x)
    # y = preprocessing.scale(y)
    print('Data Standardization Complete')

    x = preprocessing.normalize(x)
    # y = preprocessing.normalize(y)
    print('Data Normalization Complete')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    print("Shape of x_train: ", x_train.shape)
    print("Shape of y_train: ", y_train.shape)
    print("Shape of x_test: ", x_test.shape)
    print("Shape of y_test", y_test.shape)
    print('Data Sliced In Training And Testing Sets')

    print("Model Training Started")
    algorithm = "Linear"
    model = train_classifier(x_train, x_test, y_train, y_test, algorithm)
    print("Model Training Complete")

    viewer_data['Predicted_Viewership'] = model.predict(x)
    viewer_data.to_csv('./Prediction_data/predicted_file.csv')


if __name__ == '__main__':
    main('./Prediction_data/simpsons_episodes.csv')
