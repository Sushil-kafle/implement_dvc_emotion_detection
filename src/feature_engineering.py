import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

import dvc.api


params = dvc.api.params_show().get("feature_engineering", {})


train_data = pd.read_csv("data/processed/train.csv")
test_data = pd.read_csv("data/processed/test.csv")


X_train = train_data["text"].values
y_train = train_data["label"].values

X_test = test_data["text"].values
y_test = test_data["label"].values


vectorizer = CountVectorizer(max_features=params["max_features"])

X_train_bow = vectorizer.fit_transform(X_train)

X_test_bow = vectorizer.transform(X_test)


train_df = pd.DataFrame(X_train_bow.toarray())
train_df["label"] = y_train


test_df = pd.DataFrame(X_test_bow.toarray())
test_df["label"] = y_test


data_path = os.path.join("data", "features")

if not os.path.exists(data_path):
    os.makedirs(data_path)

train_df.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_df.to_csv(os.path.join(data_path, "test.csv"), index=False)
