from functools import wraps
import time
import numpy as np
import pandas as pd

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

class Preprocessor:
    def __init__(
        self,
        # 電子書籍IDとページ番号の組み合わせの絞り込み条件。値のユニーク数の quantile_contentsidpagenolist * 100 ％点以上の組み合わせに絞り込む。
        quantile_contentsidpagenolist,
        # 各処理の実行時間などを画面へ出力する場合、True。
        verbose = True
    ):
        self.quantile_contentsidpagenolist = quantile_contentsidpagenolist
        self.verbose = verbose

        self._contentsidlist = None
        self._contentsidpagenolist = None
        self._operationnamengramlist = {}

    # pickle.dump() で保存する変数を指定する。
    def __getstate__(self):
        state = {
            "quantile_contentsidpagenolist": self.quantile_contentsidpagenolist,
            "verbose": self.verbose,
            "_contentsidlist": self._contentsidlist,
            "_contentsidpagenolist": self._contentsidpagenolist,
            "_operationnamengramlist": self._operationnamengramlist
        }

        return state

    # pickle.load() で復元する変数を指定する。
    def __setstate__(self, state):
        self.quantile_contentsidpagenolist = state["quantile_contentsidpagenolist"]
        self.verbose = state["verbose"]
        self._contentsidlist = state["_contentsidlist"]
        self._contentsidpagenolist = state["_contentsidpagenolist"]
        self._operationnamengramlist = state["_operationnamengramlist"]

    # 学習データとテストデータの両方で必要になる情報をあらかじめ取得しておく。
    def fit(self, eventstream):
        eventstream = eventstream.copy()
        eventstream["operationname"] = eventstream["operationname"].str.replace(" ", "_")

        self._contentsidlist = self._fit_contentsidlist(eventstream)
        self._contentsidpagenolist = self._fit_contentsidpagenolist(eventstream, self.quantile_contentsidpagenolist)
        self._operationnamengramlist[1] = self._fit_operationnamengramlist(eventstream, 1)
        self._operationnamengramlist[2] = self._fit_operationnamengramlist(eventstream, 2)
        self._operationnamengramlist[3] = self._fit_operationnamengramlist(eventstream, 3)

        return self

    # 前処理を施す。
    def transform(self, eventstream):
        eventstream = eventstream.copy()
        eventstream["operationname"] = eventstream["operationname"].str.replace(" ", "_")

        func = [
            ("min", np.nanmin),
            ("q50", np.nanmedian),
            ("max", np.nanmax),
            ("mean", np.nanmean),
            ("std", np.nanstd),
            ("sum", np.nansum)
        ]
        func_dist = [
            ("min", np.nanmin),
            ("q05", lambda a: np.nanquantile(a, q = 0.05)),
            ("q25", lambda a: np.nanquantile(a, q = 0.25)),
            ("q50", np.nanmedian),
            ("q75", lambda a: np.nanquantile(a, q = 0.75)),
            ("q95", lambda a: np.nanquantile(a, q = 0.95)),
            ("max", np.nanmax),
            ("mean", np.nanmean),
            ("std", np.nanstd),
            ("sum", np.nansum)
        ]
        data = [
            self._transform_contentsid_dummy(eventstream, self._contentsidlist),
            self._transform_contentsid_count(eventstream, self._contentsidlist),
            self._transform_contentsidpageno_count(eventstream, self._contentsidpagenolist),
            self._transform_contentsidpagenostay_agg(eventstream, func, self._contentsidpagenolist),
            self._transform_operationnamengram_count_agg(eventstream, func, 1, self._operationnamengramlist[1]),
            self._transform_operationnamengram_count_agg(eventstream, func, 2, self._operationnamengramlist[2]),
            self._transform_operationnamengram_count_agg(eventstream, func, 3, self._operationnamengramlist[3]),
            self._transform_marker_count_agg(eventstream, func),
            self._transform_memolength_agg(eventstream, func_dist),
            self._transform_devicecode_count_agg(eventstream, func),
            self._transform_hour_count_agg(eventstream, func),
            self._transform_weekday_count_agg(eventstream, func),
            self._transform_week_count_agg(eventstream, func),
            self._transform_timedelta_agg(eventstream, func_dist),
            self._transform_timedeltaoperationname_agg(eventstream, func),
            self._transform_period(eventstream),
        ]
        data = pd.concat(data, axis = 1)
        data.fillna(0.0, inplace = True)

        # 数値変数として扱いたいため、lightgbm へ与える前に、型を float に変換しておく。
        data = data.astype(float)

        return data

    # 電子書籍IDの一覧を作る。
    @stop_watch
    def _fit_contentsidlist(self, eventstream):
        data = eventstream.copy()

        contentsidlist = data["contentsid"].unique()

        return contentsidlist

    # 電子書籍IDとページ番号の組み合わせの一覧を作る。
    # 数が多すぎるため、値のユニーク数が quantile * 100 ％点以上の組み合わせのみに絞る。
    @stop_watch
    def _fit_contentsidpagenolist(self, eventstream, quantile):
        data = eventstream.copy()

        data = (
            data.
            assign(contentsidpageno = lambda df: df["contentsid"] + "-" + df["pageno"].astype(str).str.zfill(3)).
            groupby(["userid", "contentsidpageno"]).size()
        )
        nunique = data.groupby(["contentsidpageno"]).nunique()
        th = nunique.quantile(quantile)
        contentsidpagenolist = nunique[nunique >= th].index.values

        return contentsidpagenolist

    # アクションのn-gramの一覧を作る。
    @stop_watch
    def _fit_operationnamengramlist(self, eventstream, n):
        data = eventstream.copy()

        data = self._add_operationnamengram(data, n)
        operationnamengramlist = data["operationname{}gram".format(n)].unique()

        return operationnamengramlist

    # 各電子書籍を読んだか否かを表すフラグを作成する。
    # 読んだ場合 1、読まなかった場合 0。
    @stop_watch
    def _transform_contentsid_dummy(self, eventstream, contentsidlist):
        data = eventstream.copy()

        data = data[["userid", "contentsid"]].drop_duplicates(ignore_index = True)
        data["contentsid"] = pd.Categorical(data["contentsid"], categories = contentsidlist)
        data = pd.get_dummies(data, columns = ["contentsid"])
        data = data.groupby(["userid"]).sum()

        return data

    # ユーザー別・電子書籍別のアクション回数を集計する。
    @stop_watch
    def _transform_contentsid_count(self, eventstream, contentsidlist):
        data = eventstream.copy()

        data = (
            data.
            loc[lambda df: df["contentsid"].isin(contentsidlist), :].
            set_index(["userid", "contentsid"]).
            assign(dummy = 1).
            loc[:, "dummy"]
        )
        data  = self._agg_pivot(data, "contentsid", [("size", "size")], contentsidlist)

        return data

    # ユーザー別・電子書籍別・ページ番号別のアクション回数を集計する。
    @stop_watch
    def _transform_contentsidpageno_count(self, eventstream, contentsidpagenolist):
        data = eventstream.copy()

        data = (
            data.
            assign(contentsidpageno = lambda df: df["contentsid"] + "-" + df["pageno"].astype(str).str.zfill(3)).
            loc[lambda df: df["contentsidpageno"].isin(contentsidpagenolist), :].
            set_index(["userid", "contentsidpageno"]).
            assign(dummy = 1).
            loc[:, "dummy"]
        )
        data  = self._agg_pivot(data, "contentsidpageno", [("size", "size")], contentsidpagenolist)

        return data

    # ユーザー別・電子書籍別・ページ番号別の滞在時間を集計する。
    @stop_watch
    def _transform_contentsidpagenostay_agg(self, eventstream, func, contentsidpagenolist):
        data = eventstream.copy()

        data = (
            data.
            assign(contentsidpageno = lambda df: df["contentsid"] + "-" + df["pageno"].astype(str).str.zfill(3)).
            loc[lambda df: df["contentsidpageno"].isin(contentsidpagenolist), :].
            loc[lambda df: df["operationname"].isin(["OPEN", "CLOSE", "NEXT", "PREV", "PAGE_JUMP", "SEARCH_JUMP"]), :].
            sort_values(["userid", "contentsid", "eventtime"]).
            assign(eventtimeshift = lambda df: df.groupby(["userid", "contentsid"])["eventtime"].shift(1)).
            assign(stay = lambda df: (df["eventtime"] - df["eventtimeshift"]).dt.seconds).
            set_index(["userid", "contentsidpageno"]).
            loc[:, "stay"]
        )
        data  = self._agg_pivot(data, "contentsidpageno", func, contentsidpagenolist)

        return data

    # ユーザー別・電子書籍別・アクション (n-gram) 別のアクション回数を、ユーザー別・アクション (n-gram) 別に集計する。
    @stop_watch
    def _transform_operationnamengram_count_agg(self, eventstream, func, n, operationnamengramlist):
        data = eventstream.copy()

        data = self._add_operationnamengram(data, n)
        data = (
            data.
            loc[lambda df: df["operationname{}gram".format(n)].isin(operationnamengramlist), :].
            groupby(["userid", "contentsid", "operationname{}gram".format(n)]).size()
        )
        data  = self._agg_pivot(data, "operationname{}gram".format(n), func, operationnamengramlist)

        return data

    # ユーザー別・電子書籍別・マーカー理由別のアクション回数を、ユーザー別・マーカー理由別に集計する。
    @stop_watch
    def _transform_marker_count_agg(self, eventstream, func):
        data = eventstream.copy()

        data = data.assign(marker = lambda df: df["marker"].fillna("na")).groupby(["userid", "contentsid", "marker"]).size()
        categories = ["difficult", "important", "na"]
        data  = self._agg_pivot(data, "marker", func, categories)

        return data

    # メモの長さを、ユーザー別に集計する。
    @stop_watch
    def _transform_memolength_agg(self, eventstream, func):
        data = eventstream.copy()

        data = data.groupby(["userid"])["memo_length"].agg(func)
        data.fillna(0.0, inplace = True)
        data.columns = "memolength_" + data.columns

        return data

    # ユーザー別・電子書籍別・デバイス別のアクション回数を、ユーザー別・デバイス別に集計する。
    @stop_watch
    def _transform_devicecode_count_agg(self, eventstream, func):
        data = eventstream.copy()

        data = data.groupby(["userid", "contentsid", "devicecode"]).size()
        categories = ["pc", "mobile", "tablet"]
        data  = self._agg_pivot(data, "devicecode", func, categories)

        return data

    # ユーザー別・電子書籍別・時間帯別のアクション回数を、ユーザー別・時間帯別に集計する。
    @stop_watch
    def _transform_hour_count_agg(self, eventstream, func):
        data = eventstream.copy()

        data["hour"] = data["eventtime"].dt.hour.astype(str).str.zfill(2)
        data = data.groupby(["userid", "contentsid", "hour"]).size()
        categories = ["{:02d}".format(i) for i in range(24)]
        data  = self._agg_pivot(data, "hour", func, categories)

        return data

    # ユーザー別・電子書籍別・曜日別のアクション回数を、ユーザー別・曜日別に集計する。
    @stop_watch
    def _transform_weekday_count_agg(self, eventstream, func):
        data = eventstream.copy()

        data["weekday"] = data["eventtime"].dt.isocalendar().day.astype(str)
        data = data.groupby(["userid", "contentsid", "weekday"]).size()
        categories = ["{}".format(i + 1) for i in range(7)]
        data  = self._agg_pivot(data, "weekday", func, categories)

        return data

    # ユーザー別・電子書籍別・経過週数別のアクション回数を、ユーザー別・経過週数別に集計する。
    @stop_watch
    def _transform_week_count_agg(self, eventstream, func):
        data = eventstream.copy()

        data["week"] = data["eventtime"].dt.isocalendar().week.astype(str).str.zfill(2)
        data = data.groupby(["userid", "contentsid", "week"]).size()
        categories = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17"]
        data  = self._agg_pivot(data, "week", func, categories)

        return data

    # アクションの間隔 (単位：秒) を、ユーザー別に集計する。
    @stop_watch
    def _transform_timedelta_agg(self, eventstream, func):
        data = eventstream.copy()

        data = data.sort_values(["userid", "contentsid", "eventtime"])
        data["timedelta"] = data["eventtime"] - data.groupby(["userid", "contentsid"])["eventtime"].shift(1)
        data["timedelta"] = data["timedelta"].dt.seconds
        data = data.groupby(["userid"])["timedelta"].agg(func)
        data.fillna(0.0, inplace = True)
        data.columns = "timedelta_" + data.columns

        return data

    # アクションの間隔 (単位：秒) を、ユーザー別・アクション別に集計する。
    @stop_watch
    def _transform_timedeltaoperationname_agg(self, eventstream, func):
        data = eventstream.copy()

        data = data.sort_values(["userid", "contentsid", "operationname", "eventtime"])
        data["timedelta"] = data["eventtime"] - data.groupby(["userid", "contentsid", "operationname"])["eventtime"].shift(1)
        data["timedelta"] = data["timedelta"].dt.seconds
        data.rename(columns = { "operationname": "timedeltaoperationname" }, inplace = True)
        data.set_index(["userid", "contentsid", "timedeltaoperationname"], inplace = True)
        data = data["timedelta"]
        categories = [
            "OPEN", "CLOSE",
            "NEXT", "PREV",
            "PAGE_JUMP",
            "ADD BOOKMARK", "ADD MARKER", "ADD MEMO",
            "CHANGE MEMO",
            "DELETE BOOKMARK", "DELETE MARKER", "DELETE_MEMO",
            "LINK_CLICK",
            "SEARCH", "SEARCH_JUMP"
        ]
        data  = self._agg_pivot(data, "timedeltaoperationname", func, categories)

        return data

    # データの期間を表すフラグを作成する。
    # コース開始から5週までのデータなら 3、コース開始から10週までのデータなら 2、コース開始から15週までのデータなら 1。
    @stop_watch
    def _transform_period(self, eventstream):
        data = eventstream.copy()

        tail = data["eventtime"].max()
        data["period"] = (
            # コース開始から5週までのデータなら 3
            (tail <= pd.to_datetime("2020-02-09")) * 1 +
            # コース開始から10週までのデータなら 2
            (tail <= pd.to_datetime("2020-03-15")) * 1 +
            # コース開始から15週までのデータなら 1
            1
        )
        data = data[["userid", "period"]].drop_duplicates().set_index("userid")

        return data

    # ユーザー別・指定列別に集計する。
    def _agg_pivot(self, summ, column, func, categories):
        # 集計
        agg = summ.groupby(["userid", column]).agg(func).reset_index()

        # 各集計結果を縦持ちから横持ちへ変換し、変換結果を一つのテーブルへ結合
        data = []
        for name, _ in func:
            # 縦持ちから横持ちへ変換
            data_sub = agg.pivot(index = "userid", columns = column, values = name).fillna(0.0)
            data_sub.columns = "{}_{}_".format(column, name) + data_sub.columns
            # データ内に存在しなかったカテゴリーに対応する列を追加 (必ず全カテゴリーに対応する列が存在することを保証)
            for category in categories:
                if "{}_{}_{}".format(column, name, category) not in data_sub.columns:
                    data_sub["{}_{}_{}".format(column, name, category)] = 0.0
            # 列をカテゴリーと同じ順に並び変える (列の順番が固定されていることを保証)
            data_sub = data_sub[["{}_{}_{}".format(column, name, category) for category in categories]].copy()
            # 変換結果を一つのテーブルへ結合
            data.append(data_sub)
        data = pd.concat(data, axis = 1)

        return data

    # アクションのn-gramを追加する。
    def _add_operationnamengram(self, eventstream, n):
        data = eventstream.copy()

        data = data.sort_values(["userid", "contentsid", "eventtime"])
        for i in range(n):
            data["operationnameshift{}".format(i)] = data.groupby(["userid", "contentsid"])["operationname"].shift(i)
        data = data[["userid", "contentsid"] + ["operationnameshift{}".format(i) for i in range(n)]].dropna()
        data["operationname{}gram".format(n)] = data["operationnameshift{}".format(0)]
        for i in range(1, n):
            data["operationname{}gram".format(n)] = data["operationname{}gram".format(n)] + "__" + data["operationnameshift{}".format(i)]
        
        return data
