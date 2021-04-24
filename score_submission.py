import pandas as pd
import numpy as np
import sklearn.metrics as metrics

VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']


def get_score(df_true, df_submission):
    df_submission = df_submission.sort_values('pid')
    df_true = df_true.sort_values('pid')
    task_1_scores = [metrics.roc_auc_score(df_true[entry], df_submission[entry]) for entry in TESTS]
    print('\n ### TASK 1 SCORES')
    for index, score in enumerate(task_1_scores):
        print(TESTS[index] +"\t " + str(score))
    task1 = np.mean(task_1_scores)
    task2 = metrics.roc_auc_score(df_true['LABEL_Sepsis'], df_submission['LABEL_Sepsis'])
    print('\n ### TASK 2 SCORES')
    print('LABEL_Sepsis' + '\t ' + str(task2))
    task_3_scores = [0.5 + 0.5 * np.maximum(0, metrics.r2_score(df_true[entry], df_submission[entry])) for entry in VITALS]
    print('\n ### TASK 3 SCORES')
    for index, score in enumerate(task_3_scores):
        print(VITALS[index] + "\t " + str(score))
    task3 = np.mean(task_3_scores)
    score = np.mean([task1, task2, task3])
    print('\n ### OVERALL SCORES: ')
    print(task1, task2, task3)
    return score
