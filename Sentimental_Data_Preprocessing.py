import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *

def import_tweets(filename, header = None):

    train_dataset = pd.read_csv(filename, header = header, encoding='Latin-1', usecols=range(6), low_memory=False, index_col=False)
    train_dataset.columns = ['Sentiment', 'Id', 'Date', 'Flag', 'User', 'Text']

    # for i in ['Flag','Id','User','Date']:
    #     del train_dataset[i]
    train_dataset.sentiment = train_dataset.sentiment.replace(4,1)
    train_dataset.sentiment = train_dataset.sentiment.replace(0,-1)

    return train_dataset


def preprocess_tweet(tweet):

    if type(float) == type(tweet):
        return '-'
    tweet.lower()
    tweet = re.sub('@[^\s]+',' ', tweet)

    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',tweet)

    tweet = tweet.replace( "[^a-zA-Z#]" , " " )

    #convert "#topic" to just "topic"
    # tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = re.sub('[\s]+', ' ', tweet)

    return tweet


def main(fileName) :

    print( "Sentiment Training Data Preprocessing Started" )

    train_dataset = import_tweets( fileName )
    train_dataset.Text = train_dataset.Text.fillna( value = "" )
    print( "Sentiment Training File read" )

    train_dataset['Tidy_Tweet'] = train_dataset['Text'].apply( preprocess_tweet )
    print( "Removed @Handle" )
    print( "Removed URLs" )
    print( "Removed Special Characters, Numbers, Punctuations" )
    print( "Extra White Spaces Removed" )

    train_dataset['Tidy_Tweet'] = train_dataset['Tidy_Tweet'].apply(
        lambda x : ' '.join( [w for w in x.split() if len( w ) > 2] ) )
    print( "Removed Short Words" )

    tokenized_tweet_train = train_dataset['Tidy_Tweet'].apply( lambda x : x.split() )
    print( "Tokenization Done" )

    stemmer = PorterStemmer()
    tokenized_tweet_train = tokenized_tweet_train.apply( lambda x : [stemmer.stem( i ) for i in x] )
    print( "Stemming Done" )

    lemmatizer = WordNetLemmatizer()
    tokenized_tweet_train = tokenized_tweet_train.apply( lambda x : [lemmatizer.lemmatize( i ) for i in x] )
    print( "Lammatization Done" )

    for i in range( len( tokenized_tweet_train ) ) :
        tokenized_tweet_train[i] = ' '.join( tokenized_tweet_train[i] )

    train_dataset['Tidy_Tweet'] = tokenized_tweet_train

    outputFileName = './Preprocessed_data/preprocessed_training_data.csv'
    train_dataset.to_csv( outputFileName , index = False )

    print( 'Sentiment Train Data Preprocessing Complete. Output file generated "%s".' % outputFileName )
    return outputFileName

if __name__ == '__main__':
    main("./Sentiment_training_data/sentiment_training_data.csv")