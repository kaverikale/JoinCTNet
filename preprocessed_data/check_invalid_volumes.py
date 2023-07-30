
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/preprocessed_data/imgall_Abnormality_and_Location_Labels.csv')

for ind, row in df.iterrows():
    name = row['NoteAcc_DEID']
    path = f'/Users/kaveri/backup_May2023/CT_report_generation/dataset/{name}.npz'

    try:
        ctvol = np.load(path)['ct']
    except:
        print(name)

