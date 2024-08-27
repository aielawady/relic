from glob import glob
import os
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
import re
import multiprocessing as mp
from functools import partial


def get_1st_index(files, rank):
    last_ch_ind = 100000000
    output_ind = len(files)
    for i, f in enumerate(files):
        if f"rank{rank}" in f and int(f.split("/")[-1].split("_")[1]) < last_ch_ind:
            output_ind = i
            last_ch_ind = int(f.split("/")[-1].split("_")[1])
    return output_ind


def read_csvs(path, max_length=100000000):
    files = glob(path)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    index = max([get_1st_index(files, i) for i in range(4)]) + 1
    files = files[:index]

    if len(files) > 1:
        files = sorted(
            files,
            key=lambda x: int(x.split("_")[-1].split(".")[0])
            if x.split("_")[-1].split(".")[0].isdigit()
            else 0,
        )
    data = []
    for f in files:

        df = pd.read_csv(f)
        data.append(df)
        if sum([len(x) for x in data]) >= max_length:
            break
    df = pd.concat(data, ignore_index=True)
    return df


def process_row(data, max_eps):
    _, row = data
    obs_data = row["obs"]
    tmp = {}
    for k, v in row.items():
        if k == "obs":
            continue
        try:
            tmp[k] = np.asarray(eval(v))
        except Exception:
            pass
    dones = np.nonzero(tmp["done"])[0][:max_eps]
    return tmp, dones, obs_data, [*tmp.keys(), "obs"]


def extract_episodes_data_from_df(df, max_eps=2000):
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    n_cols = len(df.keys())
    n_rows = len(df)
    data = defaultdict(lambda: np.full((n_rows, max_eps), np.nan))
    counts = np.zeros(max_eps)

    with mp.Pool(8) as p:
        func = partial(process_row, max_eps=max_eps)
        for ep_i, (tmp, dones, obs_data, row_keys) in tqdm(
            enumerate(p.imap(func, df.iterrows(), chunksize=16)), total=len(df)
        ):
            counts[: len(dones)] += 1
            for k in row_keys:
                if k == "obs":
                    if "one_hot_target_sensor" in obs_data[0]:
                        data["target"][ep_i, np.arange(len(dones))] = [
                            np.argmax(obs_data[i]["one_hot_target_sensor"])
                            for i in dones
                        ]
                else:
                    data[k][ep_i, np.arange(len(dones))] = tmp[k][dones]
    data["counts"] = counts
    return data
