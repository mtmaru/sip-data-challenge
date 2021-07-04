from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from module.preprocessor import Preprocessor
from module.estimator import Estimator

def load():
    eventstream = pd.read_csv(
        "/data/source/EventStream.csv",
        dtype = {
            "userid": str,
            "contentsid": str,
            "operationname": str,
            "pageno": int,
            "marker": str,
            "memo_length": int,
            "devicecode": str,
            "eventtime": str,
        },
        # nrows = 10000  # debug
    )
    eventstream["eventtime"] = pd.to_datetime(eventstream["eventtime"])

    quizscore = pd.read_csv(
        "/data/source/QuizScore.csv",
        dtype = {
            "userid": str,
            "score": int
        }
    )
    quizscore.set_index("userid", inplace = True)
    quizscore = quizscore["score"]

    return eventstream, quizscore

def preprocess(eventstream):
    preprocessor = Preprocessor(quantile_contentsidpagenolist = 0.90)
    preprocessor = preprocessor.fit(eventstream)

    eventstream_05w = eventstream.loc[lambda df: df["eventtime"] <= pd.to_datetime("2020-02-09"), :].copy()
    preprocessed_05w = preprocessor.transform(eventstream_05w)
    eventstream_10w = eventstream.loc[lambda df: df["eventtime"] <= pd.to_datetime("2020-03-15"), :].copy()
    preprocessed_10w = preprocessor.transform(eventstream_10w)
    eventstream_15w = eventstream.loc[lambda df: df["eventtime"] <= pd.to_datetime("2020-12-31"), :].copy()
    preprocessed_15w = preprocessor.transform(eventstream_15w)

    return preprocessor, preprocessed_05w, preprocessed_10w, preprocessed_15w

def calc_importances(preprocessed_05w, preprocessed_10w, preprocessed_15w, quizscore, usage, rep):
    estimator = Estimator(
        n_iter = 10,
        num_leaves = 32,
        min_child_samples = 32,
        n_estimators = 1000
    )
    estimator = estimator.fit(preprocessed_05w, preprocessed_10w, preprocessed_15w, quizscore)

    importances = []
    for i, model in enumerate(estimator._modellist):
        importances_i = pd.DataFrame()
        importances_i["name"] = model.feature_name_
        importances_i["importance"] = model.feature_importances_
        importances_i["usage"] = usage
        importances_i["rep"] = rep
        importances.append(importances_i)

    return importances

def train(preprocessed_05w, preprocessed_10w, preprocessed_15w, quizscore, importances):
    important_features = importances.loc[lambda df: df["diff"] > 0, "name"]
    preprocessed_05w = preprocessed_05w[important_features].copy()
    preprocessed_10w = preprocessed_10w[important_features].copy()
    preprocessed_15w = preprocessed_15w[important_features].copy()

    estimator = Estimator(
        n_iter = 10,
        num_leaves = 32,
        min_child_samples = 32,
        n_estimators = 1000
    )
    estimator = estimator.fit(preprocessed_05w, preprocessed_10w, preprocessed_15w, quizscore)

    return estimator

print("======== Loading ========")
eventstream, quizscore = load()
print("done.")

print("\n======== Preprocessing ========")
preprocessor, preprocessed_05w, preprocessed_10w, preprocessed_15w = preprocess(eventstream)
with open("/model/preprocessor.pickle", mode = "wb") as fp:
    pickle.dump(preprocessor, fp)

if Path("/model/importances.csv").exists():
    importances = pd.read_csv("/model/importances.csv")
else:
    print("\n======== Calculating feature importances with actual objective ========")
    importances = calc_importances(preprocessed_05w, preprocessed_10w, preprocessed_15w, quizscore, "actual", 0)

    for rep in range(10):
        print("\n======== Calculating feature importances with null objective (rep. {}) ========".format(rep))
        quizscore_null = quizscore.sample(frac = 1.0, random_state = rep)
        quizscore_null.index = quizscore.index
        importances = importances + calc_importances(preprocessed_05w, preprocessed_10w, preprocessed_15w, quizscore_null, "null", rep)

    print("\n======== Calculating null importances ========")
    importances = pd.concat(importances, ignore_index = True)
    importances = importances.groupby(["usage", "name"], as_index = False)["importance"].mean()
    importances = importances.pivot(index = "name", columns = "usage", values = "importance").reset_index()
    importances["diff"] = importances["actual"] - importances["null"]
    importances.sort_values(["diff"], ascending = False, inplace = True)
    importances.to_csv("/model/importances.csv", index = False, header = True)
    importances = pd.read_csv("/model/importances.csv")
    print("done.")

with open("/model/data.pickle", mode = "wb") as fp:
    data = (preprocessed_05w, preprocessed_10w, preprocessed_15w, quizscore, importances)
    pickle.dump(data, fp)

print("\n======== Training ========")
estimator = train(preprocessed_05w, preprocessed_10w, preprocessed_15w, quizscore, importances)
with open("/model/model.pickle", mode = "wb") as fp:
    pickle.dump(estimator, fp)
