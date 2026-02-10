import social_tests as test

### PHASE 1 ###

import pandas as pd
import nltk
#nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from vaderSentiment import SentimentIntensityAnalyzer

def parse_label(label):
    result = { }
    name_start = label.find(" ")
    open_paren = label.find("(")
    result["name"] = label[name_start:open_paren].strip()

    position_end = label.find(" from")
    result["position"] = label[open_paren+1:position_end].strip()

    state_start = position_end + len(" from")
    close_paren = label.find(")")
    result["state"] = label[state_start:close_paren].strip()

    return result

def get_region_from_state(state_df, state):
    row = state_df[state_df["state"] == state]
    return row.iloc[0]["region"]

end_chars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]
def find_hashtags(message):
    import string
    hashtag_list = [ ]
    for i in range(len(message)):
        if message[i] == "#":
            j = i+1
            while j < len(message) and message[j] not in end_chars:
                j = j + 1
            hashtag_list.append(message[i:j])
    return hashtag_list

def find_sentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if score < -0.1:
        return (score, "negative")
    elif score > 0.1:
        return (score, "positive")
    else:
        return (score, "neutral")

def add_columns(data, state_df):
    
    classifier = SentimentIntensityAnalyzer()
    
    names, positions, states, regions = [], [], [], []
    for label in data["label"]:
        label_result = parse_label(label)
        names.append(label_result["name"])
        positions.append(label_result["position"])
        state = label_result["state"]
        states.append(state)
        regions.append(get_region_from_state(state_df, state))
    data["name"] = names
    data["position"] = positions
    data["state"] = states
    data["region"] = regions

    hashtags, scores, sentiments = [], [], []
    for text in data["text"]:
        hashtags.append(find_hashtags(text))
        (score, category) = find_sentiment(classifier, text)
        scores.append(score)
        sentiments.append(category)
    data["hashtags"] = hashtags
    data["score"] = scores
    data["sentiment"] = sentiments

### PHASE 2 ###

def get_sentiment_quantiles(data, col_name, col_value):
    if col_name != "":
        data = data[data[col_name] == col_value]
    
    result = [ data["score"].min() ]
    result.extend(list(round(data["score"].quantile([0.25, 0.5, 0.75]),5)))
    result.append(data["score"].max())
    return result

def get_hashtag_subset(data, col_name, col_value):
    data = data[data[col_name] == col_value]
    all_hashtags = set()
    for hashtags in data["hashtags"]:
        for tag in hashtags:
            all_hashtags.add(tag)
    return all_hashtags

def get_hashtag_rates(data):
    d = { }
    for hashtags in data["hashtags"]:
        for tag in hashtags:
            if tag not in d:
                d[tag] = 0
            d[tag] += 1
    return d

def most_common_hashtags(hashtags, count):
    best_only = { }
    while len(best_only) < count:
        curr_best = None
        curr_count = 0
        for k in hashtags:
            if hashtags[k] > curr_count and k not in best_only:
                curr_best = k
                curr_count = hashtags[k]
        best_only[curr_best] = curr_count
    return best_only

def get_hashtag_sentiment(data, hashtag):
    total = 0
    count = 0
    for index, row in data.iterrows():
        hashtags = row['hashtags']
        sent = row['sentiment']
        if hashtag in hashtags:
            count += 1
            if sent == 'positive':
                total += 1
            elif sent == 'negative':
                total -= 1
    return total / count

### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    test.test_all()
    test.run()