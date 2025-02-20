import pandas as pd
import pickle
import os
from sklearn.metrics import classification_report
import json

test_data = pd.read_csv("data/features/test.csv")


X_test = test_data.drop(columns=["label"])
y_test = test_data["label"]


with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred)

output_data = {
    "classification_report": report,
}

output_path = "output"
os.makedirs(output_path, exist_ok=True)

with open(os.path.join(output_path, "predictions.json"), "w") as json_file:
    json.dump(output_data, json_file, indent=4)


print(report)
