from data import DataLoader
from hyperparams.dodge import DODGE
from learners import FeedforwardDL
from transform.transform import Transform
from concurrent.futures import ProcessPoolExecutor


directories = ["1 day", "7 days", "14 days", "30 days", "90 days", "180 days", "365 days"]
datasets = ["hadoop", "cocoon", "deeplearning", "hive", "node", "ofbiz", "qpid"]


def run(dat, time):
    data = DataLoader.from_file("./issue_close_time/" + time + "/" + dat + ".csv",
                                target="timeOpen", col_start=1)
    Transform("cfs").apply(data)
    Transform("wfo").apply(data)
    print(sum(data.y_train), "positive samples out of", len(data.y_train), "samples")

    config = {
        "n_runs": 10,
        "transforms": ["normalize", "standardize", "robust", "maxabs", "minmax"] * 30,
        "metrics": ["d2h", "pd", "pf", "auc", "conf"],
        "random": True,
        "learners": [FeedforwardDL(random={"n_layers": (2,6), "n_units": (10,20)}, verbose=0, weighted=.5, n_epochs=50)],
        "log_path": "../log/",
        "data": [data],
        "name": dat + "-" + time
    }
    for _ in range(50):
        config["learners"].append(FeedforwardDL(random={"n_layers":(2,6),"n_units":(10,20)}, verbose=0, weighted=.5, n_epochs=50))
  
    dodge = DODGE(config)
    dodge.optimize()


if __name__ == "__main__":
    for dat in datasets:
        for time in directories:
            with ProcessPoolExecutor() as executor:
                executor.submit(run, dat, time)

