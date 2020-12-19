from hpsklearn import HyperoptEstimator, any_classifier, any_text_preprocessing
from hyperopt import tpe
from raise_utils.data import TextDataLoader
from raise_utils.transform import Transform
from raise_utils.metrics import ClassificationMetrics
import sys
import time


def loss(y_true, y_pred):
    metr = ClassificationMetrics(y_true, y_pred)
    metr.add_metrics(['d2h', 'pd', 'pf'])
    d2h, pd, pf = metr.get_metrics()
    return 2. - d2h


def main():
    for dataset in ['pitsA', 'pitsB', 'pitsC', 'pitsD', 'pitsE', 'pitsF']:
        sys.stdout = open(f'./hyperopt-log/{dataset}.txt', 'w')
        for i in range(10):
            try:
                print(f'Running {dataset}')
                print('=' * 20)
                data = TextDataLoader.from_file(
                    f'../../../Dodge/data/textmining/{dataset}.txt')

                a = time.time()
                estim = HyperoptEstimator(classifier=any_classifier('clf'),
                                          preprocessing=any_text_preprocessing(
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
