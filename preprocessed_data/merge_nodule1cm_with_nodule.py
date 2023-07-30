import pandas as pd
import csv
import numpy as np
df = pd.read_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/imgall_Abnormality_and_Location_Labels.csv')
dftest = pd.read_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/imgtest_Abnormality_and_Location_Labels.csv')
dftrain = pd.read_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/imgtrain_Abnormality_and_Location_Labels.csv')
dfvalid = pd.read_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/imgvalid_Abnormality_and_Location_Labels.csv')
columns = list(df.columns)
print(len(columns))
columns.remove('NoteAcc_DEID')
new_columns = []
for column in columns:
  splits  = column.split('*')
  new_columns.append(splits[1])

new_columns = list(set(new_columns))


for column in new_columns:
    splits  = column.split('*')
    df[f'nodule*{column}'] = df.loc[:,[f'nodule*{column}', f'nodulegr1cm*{column}']].sum(axis=1)  
    df = df.drop([f'nodulegr1cm*{column}'], axis=1)

    dftest[f'nodule*{column}'] = dftest.loc[:,[f'nodule*{column}', f'nodulegr1cm*{column}']].sum(axis=1)  
    dftest = dftest.drop([f'nodulegr1cm*{column}'], axis=1)

    dftrain[f'nodule*{column}'] = dftrain.loc[:,[f'nodule*{column}', f'nodulegr1cm*{column}']].sum(axis=1)  
    dftrain = dftrain.drop([f'nodulegr1cm*{column}'], axis=1)

    dfvalid[f'nodule*{column}'] = dfvalid.loc[:,[f'nodule*{column}', f'nodulegr1cm*{column}']].sum(axis=1)  
    dfvalid = dfvalid.drop([f'nodulegr1cm*{column}'], axis=1)

df.replace(2,1)
dftest.replace(2,1)
dftrain.replace(2,1)
dfvalid.replace(2,1)

df.to_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/preprocessed_data/imgall_Abnormality_and_Location_Labels.csv', index=False)
dftest.to_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/preprocessed_data/imgtest_Abnormality_and_Location_Labels.csv', index=False)
dftrain.to_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/preprocessed_data/imgtrain_Abnormality_and_Location_Labels.csv', index=False)
dfvalid.to_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/preprocessed_data/imgvalid_Abnormality_and_Location_Labels.csv', index=False)
print(len(df.columns))