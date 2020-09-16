from raise_utils.data import DataLoader
from raise_utils.hyperparams import DODGE
from raise_utils.learners import NaiveBayes, DecisionTree, RandomForest, LogisticRegressionClassifier, SVM
from raise_utils.transform import Transform
import os

if __name__ == "__main__":
    directories = ["1 day", "7 days", "14 days",
                   "30 days", "90 days"]
    datasets = ["camel", "cloudstack", "cocoon", "hadoop",
                "deeplearning", "hive", "node", "ofbiz", "qpid"]

    for dat in datasets:
        for time in directories:
            if f'{dat}-{time}.txt' in os.listdir('../ghost-nondl/'):
                continue
            data = DataLoader.from_file("./issue_close_time/" + time + "/" + dat + ".csv",
                                        target="timeOpen", col_start=0)

            config = {
                "n_runs": 10,
                "transforms": ["normalize", "standardize", "robust", "maxabs", "minmax"] * 30,
                "metrics": ["d2h", "pd", "pf", "prec"],
                "random": True,
                "learners": [NaiveBayes(random=True), DecisionTree(random=True), RandomForest(random=True), 
                    LogisticRegressionClassifier(random=True), SVM(random=True)],
                "log_path": "../ghost-nondl/",
                "data": [data],
                "name": dat + "-" + time
            }
            for _ in range(50):
                config["learners"].extend([NaiveBayes(random=True), DecisionTree(random=True), RandomForest(random=True), 
                    LogisticRegressionClassifier(random=True), SVM(random=True)])

            dodge = DODGE(config)
            dodge.optimize()
