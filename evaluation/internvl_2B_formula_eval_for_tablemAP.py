import os
from bs4 import BeautifulSoup
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def extract_cells(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table')
    cells = []

    for table in tables:
        for row in table.find_all('tr'):
            row_cells = []
            for cell in row.find_all(['td', 'th']):
                # 获取单元格文本，去除多余的空格和换行符
                text = ' '.join(cell.stripped_strings)
                row_cells.append(text)
            if row_cells:
                cells.append(row_cells)
    return cells

def compute_cell_matches(pred_cells, gt_cells):
    # 展平单元格列表
    pred_flat = [cell for row in pred_cells for cell in row]
    gt_flat = [cell for row in gt_cells for cell in row]

    # 计算匹配
    true_positives = len(set(pred_flat) & set(gt_flat))
    false_positives = len(set(pred_flat) - set(gt_flat))
    false_negatives = len(set(gt_flat) - set(pred_flat))

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)

    return precision, recall

def process_file(args):
    filename, model_folder, gt_folder = args
    # 读取模型输出和GT文件
    with open(os.path.join(model_folder, filename), 'r', encoding='utf-8') as f:
        model_html = f.read()
    with open(os.path.join(gt_folder, filename), 'r', encoding='utf-8') as f:
        gt_html = f.read()

    # 提取单元格
    pred_cells = extract_cells(model_html)
    gt_cells = extract_cells(gt_html)

    # 计算Precision和Recall
    precision, recall = compute_cell_matches(pred_cells, gt_cells)

    return precision, recall

def evaluate_files(model_folder, gt_folder):
    model_files = os.listdir(model_folder)
    gt_files = os.listdir(gt_folder)

    # 只比较两边都存在的文件
    common_files = list(set(model_files) & set(gt_files))

    # 准备参数列表
    args_list = [(filename, model_folder, gt_folder) for filename in common_files]

    # 使用多进程池并行处理文件
    num_processes = min(cpu_count(), 8)  # 设置最大进程数，避免过载
    with Pool(processes=num_processes) as pool:
        # 使用tqdm显示进度条
        results = list(tqdm(pool.imap(process_file, args_list), total=len(args_list), desc="Evaluating"))

    # 提取结果
    precisions, recalls = zip(*results)

    # 计算平均Precision、Recall和mAP
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mAP = mean_precision  # 如果AP等于Precision

    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall:    {mean_recall:.4f}")
    print(f"mAP:            {mAP:.4f}")

# 使用示例
model_folder = '/data1/doc_data/ysh/table/830_newcombine'
gt_folder = '/data1/doc_data/ysh/backup/InternVL/InternVL_2B/Eval/eval/table_htmls'

evaluate_files(model_folder, gt_folder)
