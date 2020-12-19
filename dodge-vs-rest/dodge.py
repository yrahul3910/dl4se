from raise_utils.data import DataLoader
import time
from raise_utils.hyperparams import DODGE
from raise_utils.learners import RandomForest, DecisionTree, LogisticRegressionClassifier, NaiveBayes
from raise_utils.transform import Transform


def main():
    directories = ["1 day", "7 days", "14 days",
                   "30 days", "90 days", "180 days", "365 days"]
    datasets = ["camel", "cloudstack", "cocoon", "hadoop",
                "deeplearning", "hive", "node", "ofbiz", "qpid"]

    for dat in datasets:
        for time_ in directories:
            a = time.time()
            data = DataLoader.from_file("/Users/ryedida/PycharmProjects/raise-package/issue_close_time/" + time_ + "/" + dat + ".csv",
                                        target="timeOpen", col_start=0)
            Transform("cfs").apply(data)
            Transform("smote").apply(data)

            config = {
                "n_runs": 10,
                "transforms": ["normalize", "standardize", "robust", "maxabs", "minmax"] * 30,
                "metrics": ["d2h", "f1", "pd", "pf", "prec"],
                "random": True,
                "learners": [NaiveBayes(random=True, name='nb0'), DecisionTree(random=True, name='dt0'), LogisticRegressionClassifier(random=True, name='lr0'), RandomForest(random=True, name='rf0')],
                "log_path": "./dodge-log/",
                "data": [data],
                "name": dat + "-" + time_ + ""
            }
            for i in range(50):
                config["learners"].extend([NaiveBayes(random=True, name=f'nb{i+1}'), DecisionTree(
                    random=True, name=f'dt{i+1}'), LogisticRegressionClassifier(random=True, name=f'lr{i+1}'), RandomForest(random=True, name=f'rf{i+1}')])

            dodge = DODGE(config)
            dodge.optimize()
            b = time.time()
            file = open(f'./dodge-log/{dat}-{time_}.txt', 'a')
            print(f'Completed in {b-a} seconds.', file=file)
            file.close()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'\nCompleted in {round(end - start, 2)} seconds')
