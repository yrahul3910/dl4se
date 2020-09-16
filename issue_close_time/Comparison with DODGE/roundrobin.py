from raise_utils.data import DataLoader
from raise_utils.hyperparams import DODGE
from raise_utils.learners import FeedforwardDL
from raise_utils.transform import Transform

import os

if __name__ == "__main__":
    directories = ["1 day", "7 days", "14 days",
                   "30 days", "90 days", "180 days", "365 days"]
    datasets = ["camel", "cloudstack", "cocoon", "hadoop",
                "deeplearning", "hive", "node", "ofbiz", "qpid"]

    for time in directories:
        for dat in datasets:
            files = [f"{d}.csv" for d in datasets if d != dat]
            files.append(f"{dat}.csv")
            data = DataLoader.from_files(f"./issue_close_time/{time}/", files,
                                        target="timeOpen", col_start=0)

            config = {
                "n_runs": 10,
                "transforms": ["normalize", "standardize", "robust", "maxabs", "minmax"] * 30,
                "metrics": ["d2h", "pd", "pf", "prec"],
                "random": True,
                "learners": [FeedforwardDL(random={'n_layers': (2, 6), 'n_units': (3, 20)}, wfo=True, weighted=1.)],
                "log_path": "../ghost-roundrobin/",
                "data": [data],
                "name": dat + "-" + time
            }
            for _ in range(50):
                config["learners"].append(FeedforwardDL(random={'n_layers': (2, 6), 'n_units': (3, 20)}, wfo=True, weighted=1.))

            dodge = DODGE(config)
            dodge.optimize()
