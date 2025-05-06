from flask import Flask, request, render_template
import nltk
import re
import string
from nltk.corpus import stopwords, twitter_samples
import numpy as np
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Download required NLTK datasets
nltk.download('twitter_samples')
nltk.download('stopwords')

def process_tweet(tweet):
    stemmer = nltk.PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = nltk.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if word not in stopwords_english and word not in string.punctuation:
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)
    return tweets_clean

def build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradientDescent(x, y, theta, alpha, num_iters):
    m = x.shape[0]
    for _ in range(num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        cost = -1. / m * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
        theta -= alpha / m * np.dot(x.T, (h - y))
    return float(cost), theta

def extract_features(tweet, freqs):
    word_l = process_tweet(tweet)
    x = np.zeros((1, 3))
    x[0, 0] = 1
    for word in word_l:
        x[0, 1] += freqs.get((word, 1.0), 0)
        x[0, 2] += freqs.get((word, 0.0), 0)
    return x

def predict_tweet(tweet, freqs, theta):
    x = extract_features(tweet, freqs)
    return sigmoid(np.dot(x, theta))

def pre(sentence):
    yhat = predict_tweet(sentence, freqs, theta)
    if yhat > 0.5:
        return 'Positive sentiment'
    elif yhat == 0.5:
        return 'Neutral sentiment'
    else:
        return 'Negative sentiment'

# Load and prepare data
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

freqs = build_freqs(train_x, train_y)

X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = extract_features(train_x[i], freqs)

Y = train_y
_, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    prediction = pre(tweet)

    # Calculate accuracy (based on test data)
    y_true = test_y  # True labels for the test set
    y_pred = [pre(t) for t in test_x]  # Predicted labels for the test set

    # Debugging: Print accuracy in the terminal
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)  # This will print in your console

    # Return prediction and accuracy to the user
    return render_template('index.html', prediction=prediction, accuracy=accuracy)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
