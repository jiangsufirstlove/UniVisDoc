from ExpRate_metrics import compute_exprate,compute_exprate_1

import os
from statistics import mean

OUTPUT_dir = '/data1/doc_data/ysh/2'
GT_dir = '/home/ysh/InternVL/InternVL_2B/Eval/eval/im2k/formula_gt_txts'
preds = os.listdir(OUTPUT_dir)

exprates,error_1s,error_2s = [],[],[]
for pred in preds:
    with open(f'{OUTPUT_dir}/{pred}',encoding='utf-8') as f:
        result = f.read()

    #result = result.replace(' ', '')     
    result = result
    with open(f'{GT_dir}/{pred}',encoding='utf-8') as f:
        #gt = f.read().replace(' ', '')     
        gt = f.read()
    exprate,error_1,error_2 = compute_exprate([result],[gt])
    
    exprates.append(exprate)
    error_1s.append(error_1)
    error_2s.append(error_2)
print(mean(exprates))
print(mean(error_1s))
print(mean(error_2s))
