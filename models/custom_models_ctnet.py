#custom_models_ctnet.py
#Copyright (c) 2020 Rachel Lea Ballantyne Draelos

#MIT License

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE

import copy
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms, models, utils

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class CTNetModel(nn.Module):
    """Model for big data. ResNet18 then 3D conv then FC."""
    def __init__(self, n_outputs, n_locations):
        super(CTNetModel, self).__init__()        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        self.diseases = ['aneurysm', 'breast_surgery', 'staple', 'density', 'calcification', 'fracture', 'suture', 'soft_tissue', 'reticulation', 'consolidation', 'pericardial_thickening', 'bronchiolitis', 'opacity', 'aspiration', 'pneumothorax', 'chest_tube', 'cyst', 'atherosclerosis', 'debris', 'postsurgical', 'arthritis', 'septal_thickening', 'bronchiolectasis', 'granuloma', 'dilation_or_ectasia', 'clip', 'fibrosis', 'catheter_or_port', 'heart_failure', 'scattered_calc', 'cabg', 'transplant', 'breast_implant', 'infection', 'pericardial_effusion', 'mass', 'tracheal_tube', 'distention', 'pacemaker_or_defib', 'lucency', 'scarring', 'scattered_nod', 'tuberculosis', 'congestion', 'inflammation', 'hardware', 'atelectasis', 'interstitial_lung_disease', 'lesion', 'sternotomy', 'cardiomegaly', 'bronchiectasis', 'pneumonia', 'cavitation', 'coronary_artery_disease', 'pleural_thickening', 'hemothorax', 'bronchitis', 'lung_resection', 'nodule', 'infiltrate', 'bandlike_or_linear', 'secretion', 'hernia', 'pneumonitis', 'gi_tube', 'pulmonary_edema', 'pleural_effusion', 'groundglass', 'heart_valve_replacement', 'honeycombing', 'airspace_disease', 'lymphadenopathy', 'cancer', 'tree_in_bud', 'bronchial_wall_thickening', 'plaque', 'other_path', 'emphysema', 'deformity', 'mucous_plugging', 'stent', 'air_trapping']
        #conv input torch.Size([1,134,512,14,14])
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(134, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU())
        
        self.classifier = nn.Sequential(
            nn.Linear(16*18*5*5, 128), #7200
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs))
        self.location_classifier = nn.Sequential(
            nn.Linear(16*18*5*5 + 84, 128), #7200
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_locations))
        
    def forward(self, x):
        shape = list(x.size())
        #example shape: [1,134,3,420,420]
        #example shape: [2,134,3,420,420]
        batch_size = int(shape[0])
        x = x.view(batch_size*134,3,420,420)
        x = self.features(x)
        x = x.view(batch_size,134,512,14,14)
        x = self.reducingconvs(x)
        #output is shape [batch_size, 16, 18, 5, 5]
        x_base = x.view(batch_size, 16*18*5*5)
        x = self.classifier(x_base)
        
        x_locations = dict.fromkeys(self.diseases, None)
        for dis in self.diseases:
            x_loc = torch.cat((x,x_base),1)
            x_loc = self.location_classifier(x_loc)
            x_locations[dis] = x_loc

        return x, x_locations
