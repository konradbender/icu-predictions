import pandas as pd
from datetime import datetime

label_string = 'LABEL_BaseExcess,LABEL_Fibrinogen,LABEL_AST,LABEL_Alkalinephos,LABEL_Bilirubin_total,' \
               'LABEL_Lactate,LABEL_TroponinI,LABEL_SaO2,LABEL_Bilirubin_direct,LABEL_EtCO2,LABEL_Sepsis,' \
               'LABEL_RRate,LABEL_ABPm,LABEL_SpO2,LABEL_Heartrate'

labels = label_string.split(",")


class Submission:

    def __init__(self, test_index):
        self.data = pd.DataFrame(columns=labels, index=test_index)


    def add_task_1_dict(self, predictions):
        for label, prediction in predictions.items():
            self.data[label] = prediction
        return

    def add_task_1(self, matrix, labels_1):
        for index,label in enumerate(labels_1):
            self.data[label] = matrix[:, index]

    def add_task_2(self, vector):
        self.data['LABEL_Sepsis'] = vector

    def add_task_3(self, predictions):
        for label, prediction in predictions.items():
            self.data[label] = prediction
            
            
    def save(self):
        self.data.to_csv('our_pred.zip', index=True, float_format='%.3f', compression='zip')
        print('saved')

