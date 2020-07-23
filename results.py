import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import numpy as np
import re, string, random
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('twitter_samples')
import matplotlib.pyplot as plt



def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

if __name__ == "__main__":

    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

    stop_words = stopwords.words('english')

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    freq_dist_pos = FreqDist(all_pos_words)
    print(freq_dist_pos.most_common(10))

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, 1)
                         for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, 0)
                         for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    train_data = dataset[:7000]
    test_data = dataset[7000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))


def stock_analysis(stock_tag, max_input_days, graph=False, pred=[]):
    num_days_list = list(range(1, max_input_days + 1))
    print(stock_tag + ":")

    # read csv files
    stocks_df = pd.read_csv(stock_tag + '.csv')
    tweets_df = pd.read_csv(stock_tag + 'tweets.csv', dtype="string")

    # drop days from tweets df that are on weekends and holidays because the stock market is only open on weekdays
    tweets_dropped = pd.merge(stocks_df["Date"], tweets_df, on="Date").drop(columns=["Date"])

    # process tweets into sentiment values for a given day
    tweets_tokenized = tweets_dropped.applymap(lambda x: word_tokenize(x) if not pd.isnull(x) else x)
    tweets_noiseless = tweets_tokenized.applymap(lambda y: remove_noise(y) if not pd.isnull([y]).any() else y)
    tweets_classified = tweets_noiseless.applymap(
        lambda z: classifier.classify(dict([token, True] for token in z)) if not pd.isnull([z]).any() else z)
    sentiment = tweets_classified.mean(axis=1)

    # drop unnecessary stock price data and get preliminary features df and labels df
    prelim = pd.concat([stocks_df.drop(columns=["Date", "High", "Low", "Close"]), sentiment], axis=1)
    prelim.dropna(inplace=True)
    features_prelim = prelim[:len(prelim) - 1]
    labels_prelim = prelim["Open"]

    # splits data and appends to give multiple days of input data
    def split_data(input_list, stagger):
        value_list_final = []
        for i in range(len(input_list) - stagger + 1):
            value_list = []
            for j in range(stagger):
                value_list.extend(input_list[i + j])
            value_list_final.append(value_list)
        return (value_list_final)

    # trains model and calculates RMSE and accuracy of whether it goes up or down
    def model(features, labels, delta_percent=False):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=78)
        reg = sklearn.linear_model.LinearRegression()
        reg.fit(X_train, y_train)
        if not delta_percent:
            return (reg.score(X_test, y_test))
        else:
            pred_delta_vals = np.sign(np.array(reg.predict(features)) - np.array(labels))
            real_delta_vals = np.sign(np.diff(np.append(np.array(labels), np.array([0]))))
            numequal = 0
            for i in range(len(pred_delta_vals)):
                if pred_delta_vals[i] == real_delta_vals[i]:
                    numequal += 1
            return (numequal / len(pred_delta_vals))

    def model_pred(features, labels, pred):
        if not len(pred) == 0:
            reg = sklearn.linear_model.LinearRegression().fit(features, labels)
            return (reg.predict(pred))

    # prints results
    for i in num_days_list:
        print(str(i) + " days of input data:")
        labels = labels_prelim[i:]

        features_sentiment = np.array(split_data(np.array([sentiment[:len(sentiment) - 1]]).T, i))
        print("RSME with sentiment only: " + str(model(features_sentiment, labels)))

        features_stock = np.array(split_data(np.array(features_prelim.drop(columns=0)), i))
        print("RSME with stock only: " + str(model(features_stock, labels)))

        features_stock_and_sentiment = np.array(split_data(np.array(features_prelim), i))
        print("RSME with stock and sentiment: " + str(model(features_stock_and_sentiment, labels)))

        print("Predicts whether it goes up or down with " + str(
            model(features_stock_and_sentiment, labels, True) * 100) + "% accuracy\n")

    print("\n\n\n\n")


for i in ["HTZ"]:
  stock_analysis(i, 10, graph=True)