from md2html import md2html
from TEDS_metric import TEDS
from statistics import mean
import os


OUTPUT_DIR='/data1/doc_data/ysh/table/827_randomsplit4'
GT_DIR='/home/ysh/InternVL/InternVL_2B/Eval/eval/table_htmls'

preds = os.listdir(OUTPUT_DIR)
scores = []
for pred in preds:
    with open(f'{OUTPUT_DIR}/{pred}',encoding='utf-8') as f:
        pred_md = f.read()
    pred_md=pred_md[8:-3]
    pred_html = md2html(pred_md)

    with open(f'{GT_DIR}/table_{pred.split(".")[0].split("_")[1]}.html',encoding='utf-8') as f:
        gt = f.read()
    teds = TEDS(n_jobs=1)
    score = teds.sample_evaluate(pred_html, gt)
    print(score)
    scores.append(score)
#print(mean(scores))
print(f"平均得分: {mean(scores)}")
