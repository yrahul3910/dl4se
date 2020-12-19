from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from hyperopt import tpe
from raise_utils.data import DataLoader
from raise_utils.metrics import ClassificationMetrics
import sys
import time


def loss(y_true, y_pred):
    metr = ClassificationMetrics(y_true, y_pred)
    metr.add_metrics(['d2h', 'pd', 'pf'])
    d2h, pd, pf = metr.get_metrics()
    return 2. - d2h


def main():
    file_dic = {"ivy":     ["ivy-1.4.csv", "ivy-2.0.csv"],
                "lucene":  ["lucene-2.0.csv", "lucene-2.2.csv"],
                "lucene2": ["lucene-2.2.csv", "lucene-2.4.csv"],
                "poi":     ["poi-1.5.csv", "poi-2.5.csv"],
                "poi2": ["poi-2.5.csv", "poi-3.0.csv"],
                "synapse": ["synapse-1.0.csv", "synapse-1.1.csv"],
                "synapse2": ["synapse-1.1.csv", "synapse-1.2.csv"],
                "camel": ["camel-1.2.csv", "camel-1.4.csv"],
                "camel2": ["camel-1.4.csv", "camel-1.6.csv"],
                "xerces": ["xerces-1.2.csv", "xerces-1.3.csv"],
                "jedit": ["jedit-3.2.csv", "jedit-4.0.csv"],
                "jedit2": ["jedit-4.0.csv", "jedit-4.1.csv"],
                "log4j": ["log4j-1.0.csv", "log4j-1.1.csv"],
                "xalan": ["xalan-2.4.csv", "xalan-2.5.csv"]
                }

    for dataset in file_dic:
        sys.stdout = open(f'./hyperopt-log/{dat}.txt', 'w')
        print(f'Running {dat}')
        print('=' * 20)
        data = DataLoader.from_files(
            base_path='./issue_close_time/', files=file_dic[dataset])

        try:
            a = time.time()
            estim = HyperoptEstimator(classifier=any_classifier('clf'),
                                      preprocessing=any_preprocessing(
                'pre'),
                algo=tpe.suggest,
                max_evals=30,
                loss_fn=loss,
                trial_timeout=30)

            estim.fit(data.x_train, data.y_train)
            preds = estim.predict(data.x_test)
            metr = ClassificationMetrics(data.y_test, preds)
            metr.add_metrics(['d2h', 'pd', 'pf'])
            print(metr.get_metrics())
            print(estim.best_model())
            b = time.time()

            print('Completed in', b-a, 'seconds.')
            except:
                continue


if __name__ == '__main__':
    main()
