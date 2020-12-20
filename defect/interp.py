from raise_utils.interpret import DODGEInterpreter
import os
import numpy as np

files = os.listdir('./ghost-v2-log')
for file in files:
    print(file, end=': ')
    interp = DODGEInterpreter(
        files=[f'./ghost-v2-log/{file}'], metrics=['f1', 'pd', 'prec'])
    res = interp.interpret()
    print(round(np.median(res[file]['f1']), 3) * 100)
