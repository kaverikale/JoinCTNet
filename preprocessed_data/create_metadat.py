import pandas as pd
import csv
'''
df = pd.read_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/Summary_3630.csv')
print(df) 
columns = df.columns
df2 = pd.read_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/preprocessed_data/binary_data/imgtest_BinaryLabels.csv')

ids = list(df2['AAA'])
newdf = pd.DataFrame(columns = columns)
for id in ids:
    new_df = df.loc[df['NoteAcc_DEID'] == id]
    newdf = pd.concat([newdf, new_df], axis=0)


newdf.to_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/preprocessed_data/metadata/test_matadata.csv', index=False)
'''
df = pd.read_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/Summary_3630.csv')

ids = list(df['Accession'])
success = []
full_file_path = []
for id in ids:
    success.append('success')
    full_file_path.append(f'{id}.npz')

df["status"] = success
df["full_filename_npz"] = full_file_path

df = df.reindex(sorted(df.columns), axis=1)



df.to_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/preprocessed_data/metadata/Summary_log.csv', index=False)


    

