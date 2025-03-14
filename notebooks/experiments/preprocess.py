import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


nltk.download("wordnet")
nltk.download("stopwords")


def lemmatization(text):
    lemmatizer = WordNetLemmatizer()

    text = text.split()

    text = [lemmatizer.lemmatize(y) for y in text]

    return " ".join(text)


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)


def removing_numbers(text):
    text = "".join([i for i in text if not i.isdigit()])
    return text


def lower_case(text):
    text = text.split()

    text = [y.lower() for y in text]

    return " ".join(text)


def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = text.replace(
        "؛",
        "",
    )

    ## remove extra whitespace
    text = re.sub("\s+", " ", text)
    text = " ".join(text.split())
    return text.strip()


def removing_urls(text):
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub(r"", text)


def remove_short_sentences(df, column, min_words):
    return df[df[column].str.split().str.len() >= min_words]


def normalized_sentence(sentence):
    sentence = lower_case(sentence)
    sentence = remove_stop_words(sentence)
    sentence = removing_numbers(sentence)
    sentence = removing_punctuations(sentence)
    sentence = removing_urls(sentence)
    sentence = lemmatization(sentence)
    return sentence
