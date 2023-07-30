import pandas as pd
import csv
import numpy as np
df = pd.read_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/preprocessed_data/imgvalid_Abnormality_and_Location_Labels.csv')
columns = list(df.columns)
print(len(columns))
new_columns = []
for column in columns:
  splits  = column.split('*')
  new_columns.append(splits[0])
new_columns.remove('NoteAcc_DEID')
new_columns = ['AAA'] + list(set(new_columns))
print(len(new_columns), new_columns)
new_df = pd.DataFrame(0, index=np.arange(len(df)), columns=new_columns)
for ind, row in df.iterrows():
  for column in columns:
    if column == 'NoteAcc_DEID':
        new_df.loc[ind,'AAA'] = row['NoteAcc_DEID']
    else:
        splits  = column.split('*')
        if row[column] == 1:
            new_df.loc[ind, splits[0]] = 1
new_df = new_df.reindex(sorted(new_df.columns), axis=1)
new_df.to_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/preprocessed_data/imgvalid_Abnormality_and_Labels.csv', index=False)
