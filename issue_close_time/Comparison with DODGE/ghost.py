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

    for dat in datasets:
        for time in directories:
            if f'{dat}-{time}.txt' in os.listdir('./orig-ghost-log/'):
                continue
            data = DataLoader.from_file("/Users/ryedida/PycharmProjects/raise-package/issue_close_time/" + time + "/" + dat + ".csv",
                                        target="timeOpen", col_start=0)

            config = {
                "n_runs": 10,
                "transforms": ["normalize", "standardize", "robust", "maxabs", "minmax"] * 30,
                "metrics": ["d2h", "pd", "pf", "prec"],
                "random": True,
                "learners": [FeedforwardDL(wfo=True, weighted=True, random={'n_layers': (2, 6), 'n_units': (3, 20)}, n_epochs=20)],
                "log_path": "./orig-ghost-log/",
                "data": [data],
                "name": dat + "-" + time
            }
            for _ in range(50):
                config["learners"].append(
                    FeedforwardDL(weighted=True, wfo=True, random={'n_layers': (2, 6), 'n_units': (3, 20)}, n_epochs=20))

            dodge = DODGE(config)
            dodge.optimize()
