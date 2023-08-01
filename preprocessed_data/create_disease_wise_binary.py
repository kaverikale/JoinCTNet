import pandas as pd
import csv
import numpy as np
import json
df1 = pd.read_csv('imgtrain_Abnormality_and_Location_Labels.csv')
diseases = ['air_trapping', 'airspace_disease', 'aneurysm', 'arthritis', 'aspiration', 'atelectasis', 'atherosclerosis', 'bandlike_or_linear', 'breast_implant', 'breast_surgery', 'bronchial_wall_thickening', 'bronchiectasis', 'bronchiolectasis', 'bronchiolitis', 'bronchitis', 'cabg', 'calcification', 'cancer', 'cardiomegaly', 'catheter_or_port', 'cavitation', 'chest_tube', 'clip', 'congestion', 'consolidation', 'coronary_artery_disease', 'cyst', 'debris', 'deformity', 'density', 'dilation_or_ectasia', 'distention', 'emphysema', 'fibrosis', 'fracture', 'gi_tube', 'granuloma', 'groundglass', 'hardware', 'heart_failure', 'heart_valve_replacement', 'hemothorax', 'hernia', 'honeycombing', 'infection', 'infiltrate', 'inflammation', 'interstitial_lung_disease', 'lesion', 'lucency', 'lung_resection', 'lymphadenopathy', 'mass', 'mucous_plugging', 'nodule', 'opacity', 'other_path', 'pacemaker_or_defib', 'pericardial_effusion', 'pericardial_thickening', 'plaque', 'pleural_effusion', 'pleural_thickening', 'pneumonia', 'pneumonitis', 'pneumothorax', 'postsurgical', 'pulmonary_edema', 'reticulation', 'scarring', 'scattered_calc', 'scattered_nod', 'secretion', 'septal_thickening', 'soft_tissue', 'staple', 'stent', 'sternotomy', 'suture', 'tracheal_tube', 'transplant', 'tree_in_bud', 'tuberculosis']

locations = ['abdomen','adrenal_gland','airways','anterior','aorta','aortic_valve','axilla','bone','breast','centrilobular','chest_wall','diaphragm','esophagus','gallbladder','heart','hilum','inferior','interstitial','intestine','ivc','kidney','lateral','left','left_lower','left_lung','left_mid','left_upper','liver','lung','medial','mediastinum','mitral_valve','other_location','pancreas','posterior','pulmonary_artery','pulmonary_valve','pulmonary_vein','rib','right','right_lower','right_lung','right_mid','right_upper','spine','spleen','stomach','subpleural','superior','svc','thyroid','tricuspid_valve']

for dis in diseases:
    df = pd.DataFrame(columns = locations)
    df['AAA'] = df1['NoteAcc_DEID']
    for loc in locations:
        df[loc] = df1[f'{dis}*{loc}']

    df = df.reindex(sorted(df.columns), axis=1)
    
    df.to_csv(f'location_binary_data/imgtrain_{dis}_BinaryLabels.csv', index=False)







