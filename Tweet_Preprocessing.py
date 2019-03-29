import warnings
import ML_Sentiment
import pandas as pd
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore", category = DeprecationWarning)


def remove_http(txt):
    txt = str(txt)
    lst = list()
    for x in txt.split():
        if not x.startswith('http'):
            lst.append(x)
    return " ".join(lst)


def remove_pattern(txt,pattern):
    txt = str(txt)
    return " ".join(filter(lambda x: x[0] != pattern, txt.split()))


def main(fileName):

    print("Tweet Preprocessing Started")

    train = pd.read_csv(fileName, usecols = range(12), encoding = 'utf-8', index_col = False, low_memory = False)
    print("File Read Successful")

    train['Tidy_Tweet'] = [remove_pattern(x,'@') for x in train['Text']]
    print("Removed @Handle")

    # train['Tidy_Tweet'] = [remove_http(x) for x in train['Tidy_Tweet']]
    train['Tidy_Tweet'] = [re.sub( '((www\.[^\s]+)|(https?://[^\s]+))' , ' ' , tweet ) for tweet in train['Tidy_Tweet']]
    print("Removed URLs")

    train['Tidy_Tweet'] = train['Tidy_Tweet'].str.replace("[^a-zA-Z#]", " ")
    print("Removed Special Characters, Numbers, Punctuations")

    train['Tidy_Tweet'] = train['Tidy_Tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
    print("Removed Short Words")

    tokenized_tweet_train = train['Tidy_Tweet'].apply(lambda x : x.split())
    print("Tokenization Done")

    stemmer = PorterStemmer()
    tokenized_tweet_train = tokenized_tweet_train.apply(lambda x: [stemmer.stem(i) for i in x])
    print("Stemming Done")

    lemmatizer = WordNetLemmatizer()
    tokenized_tweet_train = tokenized_tweet_train.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
    print("Lammatization Done")

    for i in range(len(tokenized_tweet_train)):
        tokenized_tweet_train[i] = ' '.join(tokenized_tweet_train[i])

    train['Tidy_Tweet'] = tokenized_tweet_train

    outputFileName = './Preprocessed_data/tweet_preprocessed.csv'
    train.to_csv(outputFileName, index=False)

    print('Tweet Preprocessing Complete. Output file generated "%s".' % outputFileName )

    ML_Sentiment.main(outputFileName)


if __name__ == "__main__" :
    main( './Tweet_data/tweet_data.csv' )
