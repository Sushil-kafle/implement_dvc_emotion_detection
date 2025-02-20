from sklearn.naive_bayes import GaussianNB
import pickle
import os

import pandas as pd


train_data = pd.read_csv("data/features/train.csv")


X_train = train_data.drop(columns=["label"])


y_train = train_data["label"].values


model = GaussianNB()

model.fit(X_train, y_train)


if not os.path.exists("model"):
    os.makedirs("model")


with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
