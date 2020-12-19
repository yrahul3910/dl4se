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
    for dataset in ['DataClass.csv', 'FeatureEnvy.csv', 'GodClass.csv', 'LongMethod.csv']:
        sys.stdout = open(f'./hyperopt-log/{dataset}.txt', 'w')
        try:
            print(f'Running {dataset}')
            print('=' * 20)
            data = DataLoader.from_file(
                f'../../../Dodge/data/smell/{dataset}', target='SMELLS', col_start=0, col_stop=-1)

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
            print('perf:', metr.get_metrics()[0])
            print(metr.get_metrics())
            print(estim.best_model())
            b = time.time()

            print('Completed in', b-a, 'seconds.')
        except:
            continue


if __name__ == '__main__':
    main()
