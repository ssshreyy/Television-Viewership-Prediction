import pandas as pd
import bisect, datetime
import Viewership_Prediction


def date_change(str_date):
    if str_date:
        return datetime.datetime.strptime(str_date, '%d-%m-%Y').strftime('%Y-%m-%d')


def viewers_change(str_views):
    if str_views == 'NaN':
        return '0'
    return str(int(float(str_views) * 1000000))


def main(prediction_file, simpsons_file):

    print('Prediction Features Extraction Started')
    viewer_data = pd.read_csv(simpsons_file, usecols=range(13), index_col=False, low_memory = False)
    print('Episode Data File Read Successful')

    tweet_data = pd.read_csv(prediction_file, usecols=range(15), index_col=False, low_memory = False)
    print('Tweet Data File Read Successful')

    viewer_data['Air_Date'] = list(map(date_change, viewer_data['Air_Date']))
    tweet_data['Date'] = list(map(date_change, tweet_data['Date']))
    print('Date Columns Altered')

    viewer_data['US_Viewers_In_Millions'] = list(map(viewers_change, viewer_data['US_Viewers_In_Millions']))
    print('Viewer Column Altered')

    first_date = bisect.bisect_left(viewer_data['Air_Date'], '2010-01-01')
    last_date = bisect.bisect_left(viewer_data['Air_Date'], '2017-01-01')

    retweets = list()
    favorites = list()
    vaderScore = list()
    sentimentScore = list()
    tweetsPerDay = list()
    uniqueUsers = list()

    retweets.append(0)
    favorites.append(0)
    vaderScore.append(0)
    sentimentScore.append(0)
    tweetsPerDay.append(0)
    uniqueUsers.append(0)

    count = 1
    for i in range(first_date, last_date - 1):
        print(viewer_data['Air_Date'][i])
        temp1 = str(viewer_data['Air_Date'][i])
        temp2 = str(viewer_data['Air_Date'][i + 1])

        start = bisect.bisect_left(tweet_data['Date'], temp1)
        end = bisect.bisect_left(tweet_data['Date'], temp2)
        uniqueSortedDates = sorted(set(tweet_data['Date'][start:end]))

        tweetsPerDayCount = (end - start) / len(uniqueSortedDates)
        uniqueUsersCount = len(set(tweet_data['Author_ID'][start:end]))
        for i in range(start, end):
            count = uniqueSortedDates.index(str(tweet_data['Date'][i])) + 1
            retweetCount = count * float(tweet_data['Retweets'][i])
            favoriteCount = count * float(tweet_data['Favorites'][i])
            vaderCount = count * float(tweet_data['Vader_Score'][i])
            sentimentCount = count * float(tweet_data['Sentiment_Score'][i])

        retweets.append(retweetCount)
        favorites.append(favoriteCount)
        vaderScore.append(vaderCount)
        sentimentScore.append(sentimentCount)
        tweetsPerDay.append(tweetsPerDayCount)
        uniqueUsers.append(uniqueUsersCount)

    viewer_data['Retweets'] = retweets
    viewer_data['Favorites'] = favorites
    viewer_data['Vader_Score'] = vaderScore
    viewer_data['Sentiment_Score'] = sentimentScore
    viewer_data['Tweets_Per_Day'] = tweetsPerDay
    viewer_data['Unique_Users'] = uniqueUsers
    viewer_data.to_csv(simpsons_file, index = False)

    Viewership_Prediction.main(simpsons_file)


if __name__ == "__main__":
    main('./Prediction_data/tweet_predict.csv', './Prediction_data/simpsons_episodes.csv')
