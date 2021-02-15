from functools import wraps
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# 実行時間を測定するためのデコレータ。
def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        process_time =  time.time() - start
        if args[0].verbose:
            print("{}: {:.2f}s".format(func.__name__, process_time))
        return result
    return wrapper

class Estimator:
    def __init__(
        self,
        # 学習データとテストデータを変えて学習を繰り返す回数。
        n_iter,
        # LightGBMの num_leaves。
        num_leaves,
        # LightGBMの min_child_samples。
        min_child_samples,
        # LightGBMの n_estimators。
        n_estimators,
        # 学習時の予測結果を保存する場合、True。
        save_result = False,
        # RMSEなどを画面へ出力する場合、True。
        verbose = True
    ):
        self.n_iter = n_iter
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.n_estimators = n_estimators
        self.save_result = save_result
        self.verbose = verbose

        self._modellist = []
        self._result = None

    # pickle.dump() で保存する変数を指定する。
    def __getstate__(self):
        state = {
            "n_iter": self.n_iter,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "n_estimators": self.n_estimators,
            "save_result": self.save_result,
            "verbose": self.verbose,
            "_modellist": self._modellist,
            "_result": self._result
        }

        return state

    # pickle.load() で復元する変数を指定する。
    def __setstate__(self, state):
        self.n_iter = state["n_iter"]
        self.num_leaves = state["num_leaves"]
        self.min_child_samples = state["min_child_samples"]
        self.n_estimators = state["n_estimators"]
        self.verbose = state["verbose"]
        self._modellist = state["_modellist"]
        self._result = state["_result"]

    # 総合成績の予測器を学習させる。
    def fit(self, preprocessed_05w, preprocessed_10w, preprocessed_15w, quizscore):
        # 学習データとテストデータを変えて、学習を繰り返す。
        rmselist_train = []
        rmselist_05w = []
        rmselist_10w = []
        rmselist_15w = []
        rmselist_all = []
        resultlist = []
        for i in range(self.n_iter):
            model, result, rmse_train, rmse_05w, rmse_10w, rmse_15w, rmse_all = \
                self._fit_i(i, preprocessed_05w, preprocessed_10w, preprocessed_15w, quizscore)
            self._modellist.append(model)
            rmselist_train.append(rmse_train)
            rmselist_05w.append(rmse_05w)
            rmselist_10w.append(rmse_10w)
            rmselist_15w.append(rmse_15w)
            rmselist_all.append(rmse_all)
            if self.save_result:
                resultlist.append(result)
            else:
                result = None
        if self.save_result:
            self._result = pd.concat(resultlist)

        # RMSEの平均と標準偏差を出力する。
        if self.verbose:
            print("train: {:.4f}±{:.4f}, 05w: {:.4f}±{:.4f}, 10w: {:.4f}±{:.4f}, 15w: {:.4f}±{:.4f}, all: {:.4f}±{:.4f}".format(
                np.mean(rmselist_train), np.std(rmselist_train),
                np.mean(rmselist_05w), np.std(rmselist_05w),
                np.mean(rmselist_10w), np.std(rmselist_10w),
                np.mean(rmselist_15w), np.std(rmselist_15w),
                np.mean(rmselist_all), np.std(rmselist_all)
            ))

        return self

    # 総合成績を予測する。
    def predict(self, preprocessed):
        score_pred_acc = np.zeros(preprocessed.shape[0])
        for i, model in enumerate(self._modellist):
            score_pred = model.predict(preprocessed)
            score_pred_acc = score_pred_acc + score_pred
        score_pred_acc /= len(self._modellist)

        floor = lambda a: (a < 0) * 0 + (0 <= a) * a
        ceiling = lambda a: (a <= 100) * a + (100 < a) * 100
        score_pred_acc = ceiling(floor(score_pred_acc))

        return score_pred_acc

    # 総合成績の予測器を学習させる (1回分)。
    @stop_watch
    def _fit_i(self, i, preprocessed_05w, preprocessed_10w, preprocessed_15w, quizscore):
        train_preprocessed, test_preprocessed_05w, test_preprocessed_10w, test_preprocessed_15w = \
            self._fit_i_split_train_test(i, preprocessed_05w, preprocessed_10w, preprocessed_15w)
        model = self._fit_i_fit(i, train_preprocessed, quizscore)
        result = self._fit_i_predict(i, model, train_preprocessed, test_preprocessed_05w, test_preprocessed_10w, test_preprocessed_15w, quizscore)
        rmse_train, rmse_05w, rmse_10w, rmse_15w, rmse_all = self._fit_i_print_rmse(i, result)

        return model, result, rmse_train, rmse_05w, rmse_10w, rmse_15w, rmse_all

    # 学習データとテストデータに分割する。
    def _fit_i_split_train_test(self, i, preprocessed_05w, preprocessed_10w, preprocessed_15w):
        # テストデータを作成する (コース開始から5週まで)。
        test_preprocessed_05w = preprocessed_05w.sample(100, random_state = i).copy()
        # テストデータを作成する (コース開始から10週まで)。
        test_preprocessed_10w = preprocessed_10w.sample(100, random_state = i).copy()
        # テストデータを作成する (コース開始から15週まで)。
        test_preprocessed_15w = preprocessed_15w.sample(100, random_state = i).copy()
        # 学習データを作成する。
        # コース開始から5週まで、10週まで、15週までのデータを擬似的に作成し、学習データとして用いる。
        is_train = lambda df: (
            ~df.index.isin(test_preprocessed_05w.index) &
            ~df.index.isin(test_preprocessed_10w.index) &
            ~df.index.isin(test_preprocessed_15w.index)
        )
        train_preprocessed_05w = preprocessed_05w.loc[is_train, :].copy()
        train_preprocessed_10w = preprocessed_10w.loc[is_train, :].copy()
        train_preprocessed_15w = preprocessed_15w.loc[is_train, :].copy()
        train_preprocessed = pd.concat([train_preprocessed_05w, train_preprocessed_10w, train_preprocessed_15w])

        return train_preprocessed, test_preprocessed_05w, test_preprocessed_10w, test_preprocessed_15w

    # 学習データでLightGBMを学習させる。
    def _fit_i_fit(self, i, train_preprocessed, quizscore):
        model = lgb.LGBMRegressor(
            num_leaves = self.num_leaves,
            min_child_samples = self.min_child_samples,
            n_estimators = self.n_estimators,
            importance_type = "gain",
            random_state = i
        )
        model = model.fit(
            X = train_preprocessed,
            y = quizscore.loc[train_preprocessed.index],
            verbose = False
        )

        return model

    # 精度評価のため、学習データとテストデータの総合成績を予測する。
    def _fit_i_predict(self, i, model, train_preprocessed, test_preprocessed_05w, test_preprocessed_10w, test_preprocessed_15w, quizscore):
        floor = lambda a: (a < 0) * 0 + (0 <= a) * a
        ceiling = lambda a: (a <= 100) * a + (100 < a) * 100

        train_preprocessed = train_preprocessed.copy()
        train_preprocessed["score_pred"] = ceiling(floor(model.predict(train_preprocessed)))
        train_preprocessed["score_true"] = quizscore.loc[train_preprocessed.index].values
        train_preprocessed["usage"] = "train"

        test_preprocessed_05w = test_preprocessed_05w.copy()
        test_preprocessed_05w["score_pred"] = ceiling(floor(model.predict(test_preprocessed_05w)))
        test_preprocessed_05w["score_true"] = quizscore.loc[test_preprocessed_05w.index].values
        test_preprocessed_05w["usage"] = "test_05w"

        test_preprocessed_10w = test_preprocessed_10w.copy()
        test_preprocessed_10w["score_pred"] = ceiling(floor(model.predict(test_preprocessed_10w)))
        test_preprocessed_10w["score_true"] = quizscore.loc[test_preprocessed_10w.index].values
        test_preprocessed_10w["usage"] = "test_10w"

        test_preprocessed_15w = test_preprocessed_15w.copy()
        test_preprocessed_15w["score_pred"] = ceiling(floor(model.predict(test_preprocessed_15w)))
        test_preprocessed_15w["score_true"] = quizscore.loc[test_preprocessed_15w.index].values
        test_preprocessed_15w["usage"] = "test_15w"

        result = pd.concat([train_preprocessed, test_preprocessed_05w, test_preprocessed_10w, test_preprocessed_15w])
        result["i"] = i

        return result 

    # RMSEを計算し、画面に出力する。
    def _fit_i_print_rmse(self, i, result):
        rmse_train = mean_squared_error(
            result.loc[lambda df: df["usage"] == "train", "score_true"],
            result.loc[lambda df: df["usage"] == "train", "score_pred"],
            squared = False
        )
        rmse_05w = mean_squared_error(
            result.loc[lambda df: df["usage"] == "test_05w", "score_true"],
            result.loc[lambda df: df["usage"] == "test_05w", "score_pred"],
            squared = False
        )
        rmse_10w = mean_squared_error(
            result.loc[lambda df: df["usage"] == "test_10w", "score_true"],
            result.loc[lambda df: df["usage"] == "test_10w", "score_pred"],
            squared = False
        )
        rmse_15w = mean_squared_error(
            result.loc[lambda df: df["usage"] == "test_15w", "score_true"],
            result.loc[lambda df: df["usage"] == "test_15w", "score_pred"],
            squared = False
        )
        rmse_all = mean_squared_error(
            result.loc[lambda df: df["usage"] != "train", "score_true"],
            result.loc[lambda df: df["usage"] != "train", "score_pred"],
            squared = False
        )

        if self.verbose:
            print("i: {:02d}, train: {:7.4f}, 05w: {:7.4f}, 10w: {:7.4f}, 15w: {:7.4f}, all: {:7.4f}".format(
                i,
                rmse_train,
                rmse_05w, rmse_10w, rmse_15w, rmse_all
            ))

        return rmse_train, rmse_05w, rmse_10w, rmse_15w, rmse_all
