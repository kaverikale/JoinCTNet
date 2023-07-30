import pandas as pd
import csv
import numpy as np
import json
df = pd.read_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/preprocessed_data/imgall_Abnormality_and_Location_Labels.csv')
columns = list(df.columns)
print(len(columns))
new_columns = []
loc_cols = []
columns.remove('NoteAcc_DEID')

for column in columns:
  splits  = column.split('*')
  new_columns.append(splits[0])
  loc_cols.append(splits[1])

new_columns = list(set(new_columns))
loc_cols = list(set(loc_cols))
print(len(new_columns), new_columns)

disease_locations = dict.fromkeys(new_columns, [])
#for col in new_columns:
#    disease_locations[col] = []
for col in new_columns:
    for loc in loc_cols:
        if 1 in set(df[f'{col}*{loc}']):
            disease_locations[col].append(loc)
    disease_locations[col] = list(set(disease_locations[col]))
json_object = json.dumps(disease_locations,) 
with open("disease_wise_organs.json", "w") as outfile:
    json.dump(disease_locations, outfile)

print(disease_locations)

