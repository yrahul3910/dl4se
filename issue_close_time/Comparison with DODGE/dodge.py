from raise_utils.data import DataLoader
from raise_utils.hyperparams import DODGE
from raise_utils.learners import RandomForest, DecisionTree, LogisticRegressionClassifier, NaiveBayes
from raise_utils.transform import Transform

if __name__ == "__main__":
    directories = ["1 day", "7 days", "14 days",
                   "30 days", "90 days", "180 days", "365 days"]
    datasets = ["camel", "cloudstack", "cocoon", "hadoop",
                "deeplearning", "hive", "node", "ofbiz", "qpid"]

    for dat in datasets:
        for time in directories:
            data = DataLoader.from_file("/Users/ryedida/PycharmProjects/raise-package/issue_close_time/" + time + "/" + dat + ".csv",
                                        target="timeOpen", col_start=0)
            Transform("cfs").apply(data)
            Transform("smote").apply(data)

            config = {
                "n_runs": 10,
                "transforms": ["normalize", "standardize", "robust", "maxabs", "minmax"] * 30,
                "metrics": ["f1", "pd", "pf", "prec"],
                "random": True,
                "learners": [NaiveBayes(random=True, name='nb'), DecisionTree(random=True, name='nb'), LogisticRegressionClassifier(random=True, name='lr'), RandomForest(random=True, name='rf')],
                "log_path": "./dodge-log/",
                "data": [data],
                "name": dat + "-" + time + ""
            }
            for _ in range(50):
                config["learners"].extend([NaiveBayes(random=True, name='nb'), DecisionTree(
                    random=True, name='nb'), LogisticRegressionClassifier(random=True, name='lr'), RandomForest(random=True, name='rf')])

            dodge = DODGE(config)
            dodge.optimize()
