from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from module.preprocessor import Preprocessor
from module.estimator import Estimator

eventstream = pd.read_csv(
    "/data/evaluate/EventStream.csv",
    dtype = {
        "userid": str,
        "contentsid": str,
        "operationname": str,
        "pageno": int,
        "marker": str,
        "memo_length": int,
        "devicecode": str,
        "eventtime": str,
    }
)
eventstream["eventtime"] = pd.to_datetime(eventstream["eventtime"])

quizscore = pd.read_csv(
    "/data/evaluate/QuizScore.csv",
    dtype = {
        "userid": str
    }
)
userids = quizscore["userid"].unique()

with open("/model/preprocessor.pickle", mode = "rb") as fp:
    preprocessor = pickle.load(fp)
preprocessed = preprocessor.transform(eventstream)
absent_userids = set(userids) - set(preprocessed.index)
if len(absent_userids) > 0:
    preprocessed_absents = pd.DataFrame(np.zeros((len(absent_userids), preprocessed.shape[1])), index = absent_userids, columns = preprocessed.columns)
    preprocessed = preprocessed.append(preprocessed_absents)
preprocessed = preprocessed.loc[userids, :].copy()

importances = pd.read_csv("/model/importances.csv")
important_features = importances.loc[lambda df: df["diff"] > 0, "name"]
preprocessed = preprocessed[important_features].copy()

with open("/model/model.pickle", mode = "rb") as fp:
    estimator = pickle.load(fp)
out = estimator.predict(preprocessed)

out[[userid in absent_userids for userid in userids]] = 75.0

np.savetxt("/data/result/out.csv", out, fmt = "%f")
