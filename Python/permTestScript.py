import sys
# sys.path.append('../Python')  # add our python modules to the search path

import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from Structures import MultipleModel, PairMat
from RealData import RealData
from sklearn.metrics import confusion_matrix
import pandas as pd

import Permutation

sizes = [15, 20, 30, 40, 50]

modelDict = {}  # Store the models in a double dict structure
key='None'
for size in sizes:
    print(size)
    
    modelDict[size] = MultipleModel()
#         rdict[key] = None
    modelDict[size].import_from_directory(f'../Trained_models/User_models/Uniform_exits_mu67_synth/Model-{size}/{key}/Order-{key}-{size}/')
#         mdict[key].import_from_directory(f'../Trained_models/User_models/Exponential_exits_mu67_synth/Model-{size}/{key}/Order-{key}-{size}/')
#     modelDict[size] = mdict

print('Finished all model import.')

dataDict = {}
for size in sizes:
    ddict = {}
#     path_to_data = f'../Data/pubdata/HIV_1_r5/Test_{size}.mat'
    data_dir = f'../Example_Data/Uniform_exits_mu67_synth/Generated_data_{size}/TEST/'
    #data_dir = f'../Example_Data/Exponential_exits_mu67_synth/Generated_data_{size}/TEST/'
#     for key in ['None']:
    files = os.listdir(data_dir)
    
    dataDict[size] = PairMat(os.path.join(data_dir, files[0]), method='HC')
    print(size, key)
#     dataDict[size] = ddict

print('Finished all data import and sorting.')

replace_sizes = [f for f in range(2,5,1)]
num_to_test = 1000

acc_values = {}
consistency_values = {}
for sz in sizes:
    for nreplace in replace_sizes:
        print(sz, nreplace)
        acc, conn = Permutation.sample_permutation_test(data=dataDict[sz].pairwise_mats, model=modelDict[sz],
                                                        real_labels=dataDict[sz].train.long_labels - 1, num_swap=2,
                                                        num_im=num_to_test)
        acc_values[sz, nreplace] = acc
        consistency_values[sz, nreplace] = conn


flat_acc = {key: np.mean(np.asarray(value)) for key, value in acc_values.items()}
df_dict = {}
for key in sizes:
    series = {n:flat_acc[key, n] for n in replace_sizes}
    df_dict[key] = series
new_df = pd.DataFrame.from_dict(df_dict).T
with open('../images/tables/S2_acc_HC.latex', 'w') as f:
    f.write(new_df.style.to_latex())
print('Exiting script')
