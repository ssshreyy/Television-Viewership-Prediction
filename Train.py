import pandas as pd
import bisect, datetime
from sklearn import linear_model
from sklearn import neighbors


def compute(tweet_data, start, end):
    if start == end:
        return 0
    temp1 = 0
    temp2 = 0
    count = 1
    for i in range(start, end+1):
        temp2 = float(tweet_data['Score'][i]) * 1000
        if temp2 != 0:
            count += 1
            temp1 += temp2
    return float(temp1)


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


def main(fileName):

    viewer_data = pd.read_csv('./Prediction_data/simpsons_episodes.csv', index_col=False, usecols=range(13))
    tweet_data = pd.read_csv(fileName, index_col=False, usecols=range(15), low_memory = False)
    print(tweet_data['Text'])

    print('1')
    viewer_data['Air_Date'] = list(map(date_change, viewer_data['Air_Date']))
    viewer_data['US_Viewers_In_Millions'] = list(map(viewers_change, viewer_data['US_Viewers_In_Millions']))

    print('2')
    tweet_data['Date'] = list(map(date_change, tweet_data['Date']))

    first_date = bisect.bisect_left(viewer_data['Air_Date'], '2010-01-01')
    last_date = bisect.bisect_left(viewer_data['Air_Date'], '2011-01-01')

    y = list(map(int,viewer_data['US_Viewers_In_Millions'][first_date+1:last_date]))

    final_score = list()
    print('3')
    count = 1
    for i in range(first_date, last_date - 1):
        temp1 = str(viewer_data['Air_Date'][i])
        temp2 = str(viewer_data['Air_Date'][i + 1])
        temp3 = []
        start = bisect.bisect_left(tweet_data['Date'], temp1)
        end = bisect.bisect_left(tweet_data['Date'], temp2)
        temp3.append(compute(tweet_data, start, end))

        # print(count, temp2, viewer_data['Title'][i + 1], temp3)

        count += 1
        final_score.append(temp3)

    print('4')
    print(final_score)
    print(y)
    clf = neighbors.KNeighborsClassifier(8)
    clf.fit(final_score,y)

    regression = linear_model.LinearRegression()
    regression.fit(final_score, y)

    print('5')

    first_date = bisect.bisect_left(viewer_data['Air_Date'], '2014-01-01')
    last_date = bisect.bisect_left(viewer_data['Air_Date'], '2015-01-01')

    y = list(map(int, viewer_data['US_Viewers_In_Millions'][first_date + 1:last_date]))

    predict_data_score = list()
    print('6')
    count = 1
    for i in range(first_date, last_date - 1):
        temp1 = str(viewer_data['Air_Date'][i])
        temp2 = str(viewer_data['Air_Date'][i + 1])
        temp3 = []
        start = bisect.bisect_left(tweet_data['Date'], temp1)
        end = bisect.bisect_left(tweet_data['Date'], temp2)
        temp3.append(compute(tweet_data, start, end))

        # print(count, temp2, viewer_data['Title'][i + 1], temp3)

        count += 1
        predict_data_score.append(temp3)

    print('7')
    print(predict_data_score)
    print(y)
    print(regression.predict(predict_data_score))
    print(clf.predict(predict_data_score))

    acc = clf.score(predict_data_score,y)

    print(acc)

    accuracy = regression.score(predict_data_score, y)

    print(accuracy)
    print(y)


if __name__ == "__main__":
    main('./Prediction_data/tweet_2009_predict.csv')
