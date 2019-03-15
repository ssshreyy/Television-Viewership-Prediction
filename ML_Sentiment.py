import Train
import Sentimental_Data_Preprocessing
import Sentiment_Analysis


def main(fileName):

    trainFileName = './Sentiment_training_data/sentiment_training_data.csv'
    preprocessedTrainFileName = Sentimental_Data_Preprocessing.main( trainFileName )

    outputFileName = Sentiment_Analysis.main(preprocessedTrainFileName)
    episodeFileName = './Prediction_data/simpsons_episodes.csv'

    Train.main(outputFileName, episodeFileName)


if __name__ == "__main__":
    main('./Preprocessed_data/tweet-preprocessed.csv')
