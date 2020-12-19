from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from hyperopt import tpe
from raise_utils.data import DataLoader
from raise_utils.metrics import ClassificationMetrics
from functools import partial
import sys
import time


def loss(d, t, y_true, y_pred):
    metr = ClassificationMetrics(y_true, y_pred)
    metr.add_metrics(['d2h', 'pd', 'pf'])
    d2h, pd, pf = metr.get_metrics()
    file = open(f'./hyperopt-log/{d}-{t}.txt', 'a')
    print(f'd2h = {d2h}\tpd = {pd}\tpf = {pf}',
          file=file)
    return 2. - d2h


def main():

    directories = ["1 day", "7 days", "14 days",
                   "30 days", "90 days", "180 days", "365 days"]
    datasets = ["camel", "cloudstack", "cocoon", "hadoop",
                "deeplearning", "hive", "node", "ofbiz", "qpid"]

    for dat in datasets:
        for time_ in directories:
            sys.stdout = open(f'./hyperopt-log/{dat}-{time_}.txt', 'w')
            print(f'Running {dat}-{time_}')
            print('=' * 30)
            data = DataLoader.from_file("/Users/ryedida/PycharmProjects/raise-package/issue_close_time/" + time_ + "/" + dat + ".csv",
                                        target="timeOpen", col_start=0)

            try:
                a = time.time()
                estim = HyperoptEstimator(classifier=any_classifier('clf'),
                                          preprocessing=any_preprocessing(
                                              'pre'),
                                          algo=tpe.suggest,
                                          max_evals=30,
                                          loss_fn=partial(loss, dat, time_),
                                          trial_timeout=30)

                estim.fit(data.x_train, data.y_train)
                preds = estim.predict(data.x_test)
                metr = ClassificationMetrics(data.y_test, preds)
                metr.add_metrics(['d2h', 'pd', 'pf'])
                print(metr.get_metrics())
                print(estim.best_model())
                b = time.time()

                print('Completed in', b-a, 'seconds.')
            except ValueError:
                continue
            except:
                continue


if __name__ == '__main__':
    main()
