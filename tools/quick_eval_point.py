"""Quick evaluation codes for Argoverse point-wise predictions"""
import numpy as np
import torch
from eval_utils import eval_utils_point as E

data_dir = 'argoverse_results'
eval_dict = E.quick_evaluation_point_argoverse(data_dir)

for k, v in eval_dict.items():
    print('---------------------------')
    print(k)
    print('---------------------------')
    for k2, v2 in v.items():
        print(k2, v2)