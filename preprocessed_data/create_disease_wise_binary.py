import pandas as pd
import csv
import numpy as np
import json
df1 = pd.read_csv('/Users/kaveri/backup_May2023/CT_report_generation/dataset/preprocessed_data/imgvalid_Abnormality_and_Location_Labels.csv')
diseases = ['aneurysm', 'breast_surgery', 'staple', 'density', 'calcification', 'fracture', 'suture', 'soft_tissue', 'reticulation', 'consolidation', 'pericardial_thickening', 'bronchiolitis', 'opacity', 'aspiration', 'pneumothorax', 'chest_tube', 'cyst', 'atherosclerosis', 'debris', 'postsurgical', 'arthritis', 'septal_thickening', 'bronchiolectasis', 'granuloma', 'dilation_or_ectasia', 'clip', 'fibrosis', 'catheter_or_port', 'heart_failure', 'scattered_calc', 'cabg', 'transplant', 'breast_implant', 'infection', 'pericardial_effusion', 'mass', 'tracheal_tube', 'distention', 'pacemaker_or_defib', 'lucency', 'scarring', 'scattered_nod', 'tuberculosis', 'congestion', 'inflammation', 'hardware', 'atelectasis', 'interstitial_lung_disease', 'lesion', 'sternotomy', 'cardiomegaly', 'bronchiectasis', 'pneumonia', 'cavitation', 'coronary_artery_disease', 'pleural_thickening', 'hemothorax', 'bronchitis', 'lung_resection', 'nodule', 'infiltrate', 'bandlike_or_linear', 'secretion', 'hernia', 'pneumonitis', 'gi_tube', 'pulmonary_edema', 'pleural_effusion', 'groundglass', 'heart_valve_replacement', 'honeycombing', 'airspace_disease', 'lymphadenopathy', 'cancer', 'tree_in_bud', 'bronchial_wall_thickening', 'plaque', 'other_path', 'emphysema', 'deformity', 'mucous_plugging', 'stent', 'air_trapping']

locations = ['abdomen','adrenal_gland','airways','anterior','aorta','aortic_valve','axilla','bone','breast','centrilobular','chest_wall','diaphragm','esophagus','gallbladder','heart','hilum','inferior','interstitial','intestine','ivc','kidney','lateral','left','left_lower','left_lung','left_mid','left_upper','liver','lung','medial','mediastinum','mitral_valve','other_location','pancreas','posterior','pulmonary_artery','pulmonary_valve','pulmonary_vein','rib','right','right_lower','right_lung','right_mid','right_upper','spine','spleen','stomach','subpleural','superior','svc','thyroid','tricuspid_valve']

for dis in diseases:
    df = pd.DataFrame(columns = locations)
    df['AAA'] = df1['NoteAcc_DEID']
    for loc in locations:
        df[loc] = df1[f'{dis}*{loc}']

    df = df.reindex(sorted(df.columns), axis=1)
    
    df.to_csv(f'/Users/kaveri/backup_May2023/CT_report_generation/dataset/preprocessed_data/location_binary_data/imgvalid_{dis}_BinaryLabels.csv', index=False)







