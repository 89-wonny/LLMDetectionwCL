# -*- coding: utf-8 -*-


import sys
sys.version

import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from huggingface_hub import login

# Hugging Face API 토큰 입력
login(token="")

# from huggingface_hub import whoami
# print(whoami())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import requests
import time
import ast
import re
import os
import json
import random
import sklearn
# from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

# 데이터 전처리 알고리즘
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# 학습용과 검증용으로 나누는 함수
from sklearn.model_selection import train_test_split

# 교차 검증
# 지표를 하나만 설정할 경우
from sklearn.model_selection import cross_val_score
# 지표를 하나 이상 설정할 경우
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


# 평가함수
# 분류용
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, roc_curve

# 회귀용
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# 차원축소
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 군집화
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth

# 시간 측정을 위한 시간 모듈
import datetime
from tqdm import tqdm

# 형태소 벡터를 생성하기 위한 라이브러리
from sklearn.feature_extraction.text import CountVectorizer

# 형태소 벡터를 학습 벡터로 변환한다.
from sklearn.feature_extraction.text import TfidfTransformer

# 한국어 형태소 분석
# from konlpy.tag import Okt, Hannanum, Kkma, Mecab, Komoran

# 저장
import pickle

# torch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModel
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler
from transformers import get_scheduler
import argparse
from transformers import BertForMaskedLM

from google.colab import drive
drive.mount('/content/drive')

rs_sample = pd.read_csv('/content/drive/My Drive/25.1H Thesis/just_data_250423f.csv')

result_dir = f"/content/drive/My Drive/cdf/"

# 모델
# roberta-small
# roberta-base
# monologg/distilkobert
# lighthouse/mdeberta-v3-base-kor-further

tokenizer = AutoTokenizer.from_pretrained("lighthouse/mdeberta-v3-base-kor-further")
model_class = AutoModelForSequenceClassification

model_name = "lighthouse/mdeberta-v3-base-kor-further"

safe_model_name = model_name.split('/')[-1] if '/' in model_name else model_name
method = "pure_grd"

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.texts = dataframe['txt'].tolist()
        self.labels = dataframe['target'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokenized = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'input_ids': tokenized['input_ids'].squeeze(0),  # 차원 축소 유지
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)  # 데이터 타입 명확히 설정
        }


def data_preparing(data_src, tokenizer, batch_size=16, shuffle=True, return_dataset=False):
    dataset = TextDataset(data_src, tokenizer)

    if return_dataset:
        return dataset

    dataloader = DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size
    )
    return dataloader


def eval_model(
    model,
    eval_dataloader,
    device,
    num_cls=2,
    data_source=None,
    phase=None,
    wrong_predictions_list=None,
    result_file=None  # 추가
):
    total_loss = 0.0
    total_count = 0
    all_preds, all_labels, all_probs = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            loss = outputs.loss
            total_loss += loss.item() * batch['labels'].size(0)
            total_count += batch['labels'].size(0)

            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=-1)
            labels = batch['labels']

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total_count
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs) if not np.all(all_probs == all_probs[0]) else 0.0

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    tpr_at_fpr_0001 = np.interp(0.01 / 100, fpr, tpr) if len(fpr) > 0 else 0.0

    print(f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | AUROC: {auc:.4f} | TPR@FPR=0.01%: {tpr_at_fpr_0001:.4f}")

    if data_source is not None and 'txt' in data_source:
        txt_data = [text for text in data_source['txt']]
    else:
        txt_data = []

    # 결과 저장
    if result_file is not None:
        df_result = pd.DataFrame({
            'label': all_labels,
            'txt' : txt_data ,
            'pred': all_preds,
            'prob': all_probs
        })
        df_result.to_csv(result_file, index=False, encoding='utf-8-sig')

    return avg_loss, accuracy, f1, precision, recall, auc, tpr_at_fpr_0001, (fpr, tpr)



def train_model(
    model,
    train_dataloader,
    valid_dataloader,
    test_dataloader,
    device,
    learning_rate=3e-5,
    num_epochs=5,
    accumulation_steps=2,
    num_cls=2,
    run_id=0,
    early_stopping_patience=2,
    train_data=None,
    val_data=None,
    test_data=None,
    auroc_weight=0.0,
    result_file=None
):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    model.to(device)

    print(f'-+-+-+ Run {run_id+1}: Initial Validation Metrics +-+-+-')
    eval_model(model, valid_dataloader, device, num_cls, data_source=val_data, phase="initial_val")
    print('-+-+-+ END OF INITIAL EVALUATION +-+-+-')

    train_loss_log = []
    valid_loss_log = []
    best_combined_score = 0.0  # 최고 결합 점수 초기화
    best_model_state = None
    early_stop_counter = 0
    all_wrong_predictions = []
    grad_norms_log = []
    epoch_stats_log = []  # 에폭별 통계

    # 최적 성능 기록용
    best_valid_auroc = 0.0
    best_valid_tpr = 0.0

    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_count = 0
        phase_name = f"val_epoch{epoch+1}"
        valid_result_file = f"{result_dir}valid_result_run{run_id+1}_epoch{epoch+1}_{safe_model_name}_{method}.csv"
        model.train()  # 에폭 시작 시 그래디언트 초기화

        epoch_grad_norms = []

        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            loss = outputs.loss / accumulation_steps
            total_train_loss += loss.item()
            total_count += 1

            loss.backward()
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                     param_norm = p.grad.data.norm(2)
                     total_norm += param_norm.item() ** 2

            total_norm = total_norm ** 0.5
            epoch_grad_norms.append(total_norm)
            grad_norms_log.append((epoch, step, total_norm))

            if (step + 1) % accumulation_steps == 0  or (step + 1) == len(train_dataloader) :
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        epoch_avg_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0
        epoch_std_norm = np.std(epoch_grad_norms) if epoch_grad_norms else 0
        epoch_stats_log.append((epoch, epoch_avg_norm, epoch_std_norm))

        print(f"Epoch {epoch+1} - Gradient norm stats: Avg: {epoch_avg_norm:.4f}, Std: {epoch_std_norm:.4f}")

        avg_train_loss = total_train_loss / total_count
        train_loss_log.append(avg_train_loss)
        if val_data is None:
            raise ValueError("Validation data is None.")
        if test_data is None:
            raise ValueError("Test data is None.")

        print(f'Run {run_id+1}, Epoch {epoch + 1} 성능 평가:')
        # 현재 에폭에서 검증 데이터 평가 및 틀린 예측 저장
        phase_name = f"val_epoch{epoch+1}"

        avg_valid_loss, valid_acc, valid_f1, valid_precision, valid_recall, valid_auc, valid_tpr, _ = eval_model(
            model, valid_dataloader, device, num_cls,
            data_source=val_data,
            phase=phase_name,
            result_file=valid_result_file
        )
        valid_loss_log.append(avg_valid_loss)


       # AUROC와 TPR@FPR=0.01%를 결합한 점수 계산
        combined_score = auroc_weight * valid_auc + (1 - auroc_weight) * valid_tpr

        # Save the best model based on combined score
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_valid_auroc = valid_auc  # 최고 성능일 때 AUROC 값 저장
            best_valid_tpr = valid_tpr    # 최고 성능일 때 TPR 값 저장
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
            print(f"Run {run_id+1}: New best model found with combined score: {best_combined_score:.4f}")
            print(f"  - AUROC: {valid_auc:.4f} (weight: {auroc_weight:.2f})")
            print(f"  - TPR @ FPR=0.01%: {valid_tpr:.4f} (weight: {(1-auroc_weight):.2f})")
        else:
            early_stop_counter += 1
            print(f"Run {run_id+1}: No improvement. Early stop counter = {early_stop_counter}/{early_stopping_patience}")

        # Early stopping 조건 만족 시 중단
        if early_stop_counter >= early_stopping_patience:
            print(f"Run {run_id+1}: Early stopping triggered at epoch {epoch + 1}")
            break

    # 최고 성능 모델로 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Run {run_id+1}: Restored best model for final evaluation")

    # 그래디언트 노름 로그 저장
    grad_norms_df = pd.DataFrame(grad_norms_log, columns=['epoch', 'step', 'batch_norm'])
    grad_norms_df.to_csv(f"{result_dir}grad_norms_log_{run_id+1}_{safe_model_name}_{method}.csv", index=False, encoding='utf-8-sig')

    epoch_stats_df = pd.DataFrame(epoch_stats_log, columns=['epoch', 'avg_norm', 'std_norm'])
    epoch_stats_df.to_csv(f"{result_dir}epoch_grad_stats_{run_id+1}_{safe_model_name}_{method}.csv", index=False, encoding='utf-8-sig')

    print(f'----- Run {run_id+1}: Final Evaluation on Test Data -----')
    test_result_file = f"{result_dir}test_result_run{run_id+1}_{safe_model_name}_{method}.csv"
    test_loss, test_acc, test_f1, test_precision, test_recall, test_auc, test_tpr, _ = eval_model(
        model, test_dataloader, device, num_cls,
        data_source=test_data,
        phase="test",
        result_file=test_result_file
    )
    loss_log_df = pd.DataFrame({'train_loss_log': train_loss_log,'valid_loss_log': valid_loss_log})
    loss_log_df.to_csv(f"{result_dir}loss_log_{run_id+1}_{safe_model_name}_{method}.csv", index=False, encoding='utf-8')


    return train_loss_log, valid_loss_log, test_loss, test_acc, test_f1, test_precision, test_recall, test_auc, test_tpr, best_model_state, best_valid_auroc, best_valid_tpr, best_combined_score

# seed 추가
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True) # 추가


def run_multiple_experiments(
    model_class,
    model_name,
    train_data,
    val_data,
    test_data,
    tokenizer,
    device,
    num_runs=3,
    learning_rate=3e-5,
    num_epochs=5,
    batch_size=16,
    accumulation_steps=2,
    num_cls=2,
    seed=42,
    auroc_weight=0.0  # AUROC 가중치 추가
):
    # seed=42 설정
    set_seed(seed)

    # 결과 저장을 위한 리스트
    all_test_metrics = []
    best_overall_score = 0.0  # 전체 실험에서 최고 결합 점수
    best_overall_auroc = 0.0  # 최고 모델의 AUROC
    best_overall_tpr = 0.0    # 최고 모델의 TPR
    best_overall_model_state = None
    best_run_id = -1

    # 데이터로더 생성
    train_dataset = data_preparing(train_data, tokenizer, batch_size=batch_size, shuffle=True, return_dataset=True)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_dataloader = data_preparing(val_data, tokenizer, batch_size=batch_size, shuffle=False)
    test_dataloader = data_preparing(test_data, tokenizer, batch_size=batch_size, shuffle=False)

    print(f"\nUsing combined metric with AUROC weight: {auroc_weight:.2f}, TPR@FPR=0.01% weight: {(1-auroc_weight):.2f}")

    for run in range(num_runs):
        print(f"\n===== Starting Run {run+1}/{num_runs} =====")

        # 새로운 모델 인스턴스 생성
        model = model_class.from_pretrained(model_name, num_labels=num_cls)

        # 모델 학습 - 원본 데이터 전달
        _, _, test_loss, test_acc, test_f1, test_precision, test_recall, test_auc, test_tpr, best_model_state, best_valid_auroc, best_valid_tpr, best_combined_score= train_model(
            model=model,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            test_dataloader=test_dataloader,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            accumulation_steps=accumulation_steps,
            num_cls=num_cls,
            run_id=run
        )

        # 테스트 메트릭 저장
        run_metrics = {
            'loss': test_loss,
            'accuracy': test_acc,
            'f1': test_f1,
            'precision': test_precision,
            'recall': test_recall,
            'auc': test_auc,
            'tpr_at_fpr_0001': test_tpr,
            'valid_auroc': best_valid_auroc,
            'valid_tpr': best_valid_tpr,
            'combined_score': best_combined_score
        }
        all_test_metrics.append(run_metrics)

        # Update best overall model based on combined score
        if best_combined_score > best_overall_score:
            best_overall_score = best_combined_score
            best_overall_auroc = best_valid_auroc
            best_overall_tpr = best_valid_tpr
            best_overall_model_state = best_model_state
            best_run_id = run

            # best model 저장
            torch.save(best_model_state, f"{result_dir}best_model_run_{run+1}_score_{best_combined_score:.4f}_auroc_{best_valid_auroc:.4f}_tpr_{best_valid_tpr:.4f}_{safe_model_name}_{method}.pt")
            print(f"New overall best model saved from run {run+1}")
            print(f"  - Combined score: {best_combined_score:.4f}")
            print(f"  - AUROC: {best_valid_auroc:.4f}")
            print(f"  - TPR @ FPR=0.01%: {best_valid_tpr:.4f}")

    # 평균 성능 계산
    avg_metrics = {metric: np.mean([run_metric[metric] for run_metric in all_test_metrics]) for metric in all_test_metrics[0].keys()}
    std_metrics = {metric: np.std([run_metric[metric] for run_metric in all_test_metrics]) for metric in all_test_metrics[0].keys()}

    print("\n===== Average Test Results Across All Runs =====")
    print(f"Avg Test Loss: {avg_metrics['loss']:.4f} ± {std_metrics['loss']:.4f}")
    print(f"Avg Test Accuracy: {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print(f"Avg Test F1 Score: {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    print(f"Avg Test Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"Avg Test Recall: {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"Avg Test AUROC: {avg_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")
    print(f"Avg Test TPR @ FPR=0.01%: {avg_metrics['tpr_at_fpr_0001']:.4f} ± {std_metrics['tpr_at_fpr_0001']:.4f}")
    print(f"Avg Validation AUROC: {avg_metrics['valid_auroc']:.4f} ± {std_metrics['valid_auroc']:.4f}")
    print(f"Avg Validation TPR @ FPR=0.01%: {avg_metrics['valid_tpr']:.4f} ± {std_metrics['valid_tpr']:.4f}")
    print(f"Avg Combined Score: {avg_metrics['combined_score']:.4f} ± {std_metrics['combined_score']:.4f}")

    print(f"\n===== Best Model from Run {best_run_id+1} =====")
    print(f"Combined Score: {best_overall_score:.4f}")
    print(f"Best Validation AUROC: {best_overall_auroc:.4f}")
    print(f"Best Validation TPR: {best_overall_tpr:.4f}")
    print(f"Saved as: best_model_run_{best_run_id+1}_score_{best_overall_score:.4f}_auroc_{best_overall_auroc:.4f}_tpr_{best_overall_tpr:.4f}_{safe_model_name}_{method}.pt")

    # 평균 성능 저장
    summary_df = pd.DataFrame({
      'metric': list(avg_metrics.keys()),
      'mean': [avg_metrics[m] for m in avg_metrics],
      'std': [std_metrics[m] for m in std_metrics]
    })
    summary_file_path = os.path.join(result_dir, f"summary_metrics_{safe_model_name}_{method}.csv")
    summary_df.to_csv(summary_file_path, index=False, encoding='utf-8')

    # best overall model 저장
    torch.save(best_overall_model_state, f"{result_dir}best_model_overall_{safe_model_name}_{method}.pt")
    print(f"Best overall model saved as: best_model_overall_{safe_model_name}_{method}.pt")

    return all_test_metrics, avg_metrics, std_metrics, best_overall_model_state, best_run_id


def main():
   # 데이터 분할 (7:1.5:1.5)

    prompt_list = rs_sample['info.essay_prompt'].unique().tolist()
    train_prompts, temp_prompts = train_test_split(prompt_list,
                                                   test_size=0.3,
                                                   random_state=42,
                                                   )
    val_prompts, test_prompts = train_test_split(temp_prompts,
                                                 test_size=0.5,
                                                 random_state=42,
                                                 )
    train_data = rs_sample[rs_sample['info.essay_prompt'].isin(train_prompts)].reset_index(drop=True)
    val_data = rs_sample[rs_sample['info.essay_prompt'].isin(val_prompts)].reset_index(drop=True)
    test_data = rs_sample[rs_sample['info.essay_prompt'].isin(test_prompts)].reset_index(drop=True)



    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    auroc_weight = 0.0  # 이 값을 조정하여 AUROC의 중요도 설정 (0~1 사이 값)

    all_test_metrics, avg_metrics, std_metrics, best_model_state, best_run_id = run_multiple_experiments(
        model_class=model_class,
        model_name=model_name,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        tokenizer=tokenizer,
        device=device,
        num_runs=1,
        learning_rate=3e-5,
        num_epochs=5,
        batch_size=16,
        accumulation_steps=2,
        seed=42,
        auroc_weight=auroc_weight  # AUROC 가중치 전달
    )

    # Final evaluation with best model
    print("\n===== Final Evaluation with Best Overall Model =====")
    final_model = model_class.from_pretrained(model_name, num_labels=2)
    final_model.load_state_dict(best_model_state)
    final_model.to(device)

    test_dataloader = data_preparing(test_data, tokenizer, batch_size=16, shuffle=False)
    _, test_acc, test_f1, test_precision, test_recall, test_auc, test_tpr,_ = eval_model(final_model, test_dataloader, device)


start_time = time.time()

if __name__ == "__main__":
    main()

elapsed_time_sec = time.time() - start_time
elapsed_time_min = elapsed_time_sec / 60
print(f"소요 시간: {elapsed_time_min:.2f}분")

# static

safe_model_name = model_name.split('/')[-1] if '/' in model_name else model_name
method = "static_grd" 

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.texts = dataframe['txt'].tolist()
        self.labels = dataframe['target'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokenized = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'input_ids': tokenized['input_ids'].squeeze(0),  # 차원 축소 유지
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)  # 데이터 타입 명확히 설정
        }

def re_weighting_sequence(target_series, mode='static'):
    wrong_cnt = np.sum(target_series)*1.0
    right_cnt = len(target_series)*1.0 - wrong_cnt
    if wrong_cnt == 0:
        wrong_cnt = 1
    if mode == 'adaptive':
        res_series = [right_cnt*1.0 / wrong_cnt if e == 1 else 1 for e in target_series]
    elif mode == 'static':
        res_series = [2 if e == 1 else 1 for e in target_series]

    return res_series

def wrong_first_dataloader(model, tokenized_datasets, batch_size=16, num_cls=2, current_epoch=1, device=torch.device("cpu")):
    if current_epoch == 1:
        return DataLoader(tokenized_datasets, shuffle=True, batch_size=batch_size)

    eval_dataloader = DataLoader(tokenized_datasets, shuffle=False, batch_size=1)

    all_preds = []
    all_labels = []
    right_wrong_list = []

    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )


        predictions = torch.argmax(outputs.logits, dim=-1)
        all_preds.append(predictions.item())
        all_labels.append(batch["labels"].item())

        right_wrong_list.append(1 if predictions.item() != batch["labels"].item() else 0)

    # 샘플링 가중치 설정
    weights = re_weighting_sequence(right_wrong_list)

    # 가중치 길이가 맞는지 확인
    if len(weights) != len(tokenized_datasets):
        print(f"Warning: weights length {len(weights)} does not match dataset size {len(tokenized_datasets)}")

    sampler = WeightedRandomSampler(weights=weights, num_samples=len(tokenized_datasets), replacement=True)

    return DataLoader(tokenized_datasets, batch_size=batch_size, sampler=sampler)




def data_preparing(data_src, tokenizer, batch_size=16, shuffle=True, return_dataset=False):
    dataset = TextDataset(data_src, tokenizer)

    if return_dataset:
        return dataset

    dataloader = DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size
    )
    return dataloader


def eval_model(
    model,
    eval_dataloader,
    device,
    num_cls=2,
    data_source=None,
    phase=None,
    wrong_predictions_list=None,
    result_file=None  # 추가
):
    total_loss = 0.0
    total_count = 0
    all_preds, all_labels, all_probs = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            loss = outputs.loss
            total_loss += loss.item() * batch['labels'].size(0)
            total_count += batch['labels'].size(0)

            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=-1)
            labels = batch['labels']

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total_count
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs) if not np.all(all_probs == all_probs[0]) else 0.0

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    tpr_at_fpr_0001 = np.interp(0.01 / 100, fpr, tpr) if len(fpr) > 0 else 0.0

    print(f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | AUROC: {auc:.4f} | TPR@FPR=0.01%: {tpr_at_fpr_0001:.4f}")

    if data_source is not None and 'txt' in data_source:
        txt_data = [text for text in data_source['txt']]
    else:
        txt_data = []

    # 결과 저장
    if result_file is not None:
        df_result = pd.DataFrame({
            'label': all_labels,
            'txt' : txt_data ,
            'pred': all_preds,
            'prob': all_probs
        })
        df_result.to_csv(result_file, index=False, encoding='utf-8-sig')

    return avg_loss, accuracy, f1, precision, recall, auc, tpr_at_fpr_0001, (fpr, tpr)


def train_model(
    model,
    train_dataloader,
    valid_dataloader,
    test_dataloader,
    device,
    learning_rate=3e-5,
    num_epochs=5,
    accumulation_steps=2,
    num_cls=2,
    run_id=0,
    early_stopping_patience=2,
    train_data=None,
    val_data=None,
    test_data=None,
    auroc_weight=0.0,
    result_file=None
):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    model.to(device)
    print(f'-+-+-+ Run {run_id+1}: Initial Validation Metrics +-+-+-')
    eval_model(model, valid_dataloader, device, num_cls)
    print('-+-+-+ END OF INITIAL EVALUATION +-+-+-')


    train_loss_log = []
    valid_loss_log = []
    best_combined_score = 0.0  # 최고 결합 점수 초기화
    best_model_state = None
    early_stop_counter = 0
    all_wrong_predictions = []
    grad_norms_log = []
    epoch_stats_log = []  # 에폭별 통계

    # 최적 성능 기록용
    best_valid_auroc = 0.0
    best_valid_tpr = 0.0

    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_count = 0
        phase_name = f"val_epoch{epoch+1}"
        valid_result_file = f"{result_dir}valid_result_run{run_id+1}_epoch{epoch+1}_{safe_model_name}_{method}.csv"
        model.train()  # 에폭 시작 시 그래디언트 초기화
        epoch_grad_norms = []

        train_dataloader_current = wrong_first_dataloader(
            model,
            train_dataloader.dataset,
            batch_size=train_dataloader.batch_size,
            num_cls=num_cls,
            current_epoch=epoch+1,
            device=device
        )

        for step, batch in enumerate(train_dataloader_current) :
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            loss = outputs.loss / accumulation_steps
            total_train_loss += loss.item()
            total_count += 1

            loss.backward()
            total_norm = 0

            for p in model.parameters():
                if p.grad is not None:
                     param_norm = p.grad.data.norm(2)
                     total_norm += param_norm.item() ** 2

            total_norm = total_norm ** 0.5
            epoch_grad_norms.append(total_norm)
            grad_norms_log.append((epoch, step, total_norm))

            if (step + 1) % accumulation_steps == 0  or (step + 1) == len(train_dataloader) :
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        epoch_avg_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0
        epoch_std_norm = np.std(epoch_grad_norms) if epoch_grad_norms else 0
        epoch_stats_log.append((epoch, epoch_avg_norm, epoch_std_norm))

        print(f"Epoch {epoch+1} - Gradient norm stats: Avg: {epoch_avg_norm:.4f}, Std: {epoch_std_norm:.4f}")

        avg_train_loss = total_train_loss / total_count
        train_loss_log.append(avg_train_loss)
        if val_data is None:
            raise ValueError("Validation data is None.")
        if test_data is None:
            raise ValueError("Test data is None.")

        print(f'Run {run_id+1}, Epoch {epoch + 1} 성능 평가:')
        # 현재 에폭에서 검증 데이터 평가 및 틀린 예측 저장
        phase_name = f"val_epoch{epoch+1}"


        avg_valid_loss, valid_acc, valid_f1, valid_precision, valid_recall, valid_auc, valid_tpr, _ = eval_model(
            model, valid_dataloader, device, num_cls,
            data_source=val_data,
            phase=phase_name,
            result_file=valid_result_file
        )
        valid_loss_log.append(avg_valid_loss)



       # AUROC와 TPR@FPR=0.01%를 결합한 점수 계산
        combined_score = auroc_weight * valid_auc + (1 - auroc_weight) * valid_tpr

        # Save the best model based on combined score
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_valid_auroc = valid_auc  # 최고 성능일 때 AUROC 값 저장
            best_valid_tpr = valid_tpr    # 최고 성능일 때 TPR 값 저장
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
            print(f"Run {run_id+1}: New best model found with combined score: {best_combined_score:.4f}")
            print(f"  - AUROC: {valid_auc:.4f} (weight: {auroc_weight:.2f})")
            print(f"  - TPR @ FPR=0.01%: {valid_tpr:.4f} (weight: {(1-auroc_weight):.2f})")
        else:
            early_stop_counter += 1
            print(f"Run {run_id+1}: No improvement. Early stop counter = {early_stop_counter}/{early_stopping_patience}")

        # Early stopping 조건 만족 시 중단
        if early_stop_counter >= early_stopping_patience:
            print(f"Run {run_id+1}: Early stopping triggered at epoch {epoch + 1}")
            break

    # 최고 성능 모델로 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Run {run_id+1}: Restored best model for final evaluation")

    # 그래디언트 노름 로그 저장
    grad_norms_df = pd.DataFrame(grad_norms_log, columns=['epoch', 'step', 'batch_norm'])
    grad_norms_df.to_csv(f"{result_dir}grad_norms_log_{run_id+1}_{safe_model_name}_{method}.csv", index=False, encoding='utf-8-sig')

    epoch_stats_df = pd.DataFrame(epoch_stats_log, columns=['epoch', 'avg_norm', 'std_norm'])
    epoch_stats_df.to_csv(f"{result_dir}epoch_grad_stats_{run_id+1}_{safe_model_name}_{method}.csv", index=False, encoding='utf-8-sig')


    print(f'----- Run {run_id+1}: Final Evaluation on Test Data -----')
    test_result_file = f"{result_dir}test_result_run{run_id+1}_{safe_model_name}_{method}.csv"
    test_loss, test_acc, test_f1, test_precision, test_recall, test_auc, test_tpr, _ = eval_model(
        model, test_dataloader, device, num_cls,
        data_source=test_data,
        phase="test",
        result_file=test_result_file
    )
    loss_log_df = pd.DataFrame({'train_loss_log': train_loss_log,'valid_loss_log': valid_loss_log})
    loss_log_df.to_csv(f"{result_dir}loss_log_{run_id+1}_{safe_model_name}_{method}.csv", index=False, encoding='utf-8')


    return train_loss_log, valid_loss_log, test_loss, test_acc, test_f1, test_precision, test_recall, test_auc, test_tpr, best_model_state, best_valid_auroc, best_valid_tpr, best_combined_score

# seed 추가
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True) # 추가


def run_multiple_experiments(
    model_class,
    model_name,
    train_data,
    val_data,
    test_data,
    tokenizer,
    device,
    num_runs=3,
    learning_rate=3e-5,
    num_epochs=5,
    batch_size=16,
    accumulation_steps=2,
    num_cls=2,
    seed=42,
    auroc_weight=0.0  # AUROC 가중치 추가
):
    # seed=42 설정
    set_seed(seed)

    # 결과 저장을 위한 리스트
    all_test_metrics = []
    best_overall_score = 0.0  # 전체 실험에서 최고 결합 점수
    best_overall_auroc = 0.0  # 최고 모델의 AUROC
    best_overall_tpr = 0.0    # 최고 모델의 TPR
    best_overall_model_state = None
    best_run_id = -1

    # 데이터로더 생성
    train_dataset = data_preparing(train_data, tokenizer, batch_size=batch_size, shuffle=True, return_dataset=True)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_dataloader = data_preparing(val_data, tokenizer, batch_size=batch_size, shuffle=False)
    test_dataloader = data_preparing(test_data, tokenizer, batch_size=batch_size, shuffle=False)

    print(f"\nUsing combined metric with AUROC weight: {auroc_weight:.2f}, TPR@FPR=0.01% weight: {(1-auroc_weight):.2f}")

    for run in range(num_runs):
        print(f"\n===== Starting Run {run+1}/{num_runs} =====")

        # 새로운 모델 인스턴스 생성
        model = model_class.from_pretrained(model_name, num_labels=num_cls)

        # 모델 학습 - 원본 데이터 전달
        _, _, test_loss, test_acc, test_f1, test_precision, test_recall, test_auc, test_tpr, best_model_state, best_valid_auroc, best_valid_tpr, best_combined_score= train_model(
            model=model,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            test_dataloader=test_dataloader,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            accumulation_steps=accumulation_steps,
            num_cls=num_cls,
            run_id=run
        )

        # 테스트 메트릭 저장
        run_metrics = {
            'loss': test_loss,
            'accuracy': test_acc,
            'f1': test_f1,
            'precision': test_precision,
            'recall': test_recall,
            'auc': test_auc,
            'tpr_at_fpr_0001': test_tpr,
            'valid_auroc': best_valid_auroc,
            'valid_tpr': best_valid_tpr,
            'combined_score': best_combined_score
        }
        all_test_metrics.append(run_metrics)

        # Update best overall model based on combined score
        if best_combined_score > best_overall_score:
            best_overall_score = best_combined_score
            best_overall_auroc = best_valid_auroc
            best_overall_tpr = best_valid_tpr
            best_overall_model_state = best_model_state
            best_run_id = run

            # best model 저장
            torch.save(best_model_state, f"{result_dir}best_model_run_{run+1}_score_{best_combined_score:.4f}_auroc_{best_valid_auroc:.4f}_tpr_{best_valid_tpr:.4f}_{safe_model_name}_{method}.pt")
            print(f"New overall best model saved from run {run+1}")
            print(f"  - Combined score: {best_combined_score:.4f}")
            print(f"  - AUROC: {best_valid_auroc:.4f}")
            print(f"  - TPR @ FPR=0.01%: {best_valid_tpr:.4f}")

    # 평균 성능 계산
    avg_metrics = {metric: np.mean([run_metric[metric] for run_metric in all_test_metrics]) for metric in all_test_metrics[0].keys()}
    std_metrics = {metric: np.std([run_metric[metric] for run_metric in all_test_metrics]) for metric in all_test_metrics[0].keys()}

    print("\n===== Average Test Results Across All Runs =====")
    print(f"Avg Test Loss: {avg_metrics['loss']:.4f} ± {std_metrics['loss']:.4f}")
    print(f"Avg Test Accuracy: {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print(f"Avg Test F1 Score: {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    print(f"Avg Test Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"Avg Test Recall: {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"Avg Test AUROC: {avg_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")
    print(f"Avg Test TPR @ FPR=0.01%: {avg_metrics['tpr_at_fpr_0001']:.4f} ± {std_metrics['tpr_at_fpr_0001']:.4f}")
    print(f"Avg Validation AUROC: {avg_metrics['valid_auroc']:.4f} ± {std_metrics['valid_auroc']:.4f}")
    print(f"Avg Validation TPR @ FPR=0.01%: {avg_metrics['valid_tpr']:.4f} ± {std_metrics['valid_tpr']:.4f}")
    print(f"Avg Combined Score: {avg_metrics['combined_score']:.4f} ± {std_metrics['combined_score']:.4f}")

    print(f"\n===== Best Model from Run {best_run_id+1} =====")
    print(f"Combined Score: {best_overall_score:.4f}")
    print(f"Best Validation AUROC: {best_overall_auroc:.4f}")
    print(f"Best Validation TPR: {best_overall_tpr:.4f}")
    print(f"Saved as: best_model_run_{best_run_id+1}_score_{best_overall_score:.4f}_auroc_{best_overall_auroc:.4f}_tpr_{best_overall_tpr:.4f}_{safe_model_name}_{method}.pt")

    # 평균 성능 저장
    summary_df = pd.DataFrame({
      'metric': list(avg_metrics.keys()),
      'mean': [avg_metrics[m] for m in avg_metrics],
      'std': [std_metrics[m] for m in std_metrics]
    })
    summary_file_path = os.path.join(result_dir, f"summary_metrics_{safe_model_name}_{method}.csv")
    summary_df.to_csv(summary_file_path, index=False, encoding='utf-8')

    # best overall model 저장
    torch.save(best_overall_model_state, f"{result_dir}best_model_overall_{safe_model_name}_{method}.pt")
    print(f"Best overall model saved as: best_model_overall_{safe_model_name}_{method}.pt")

    return all_test_metrics, avg_metrics, std_metrics, best_overall_model_state, best_run_id


def main():
   # 데이터 분할 (7:1.5:1.5)

    prompt_list = rs_sample['info.essay_prompt'].unique().tolist()
    train_prompts, temp_prompts = train_test_split(prompt_list,
                                                   test_size=0.3,
                                                   random_state=42,
                                                   )
    val_prompts, test_prompts = train_test_split(temp_prompts,
                                                 test_size=0.5,
                                                 random_state=42,
                                                 )
    train_data = rs_sample[rs_sample['info.essay_prompt'].isin(train_prompts)].reset_index(drop=True)
    val_data = rs_sample[rs_sample['info.essay_prompt'].isin(val_prompts)].reset_index(drop=True)
    test_data = rs_sample[rs_sample['info.essay_prompt'].isin(test_prompts)].reset_index(drop=True)


    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    auroc_weight = 0.0  # 이 값을 조정하여 AUROC의 중요도 설정 (0~1 사이 값)

    all_test_metrics, avg_metrics, std_metrics, best_model_state, best_run_id = run_multiple_experiments(
        model_class=model_class,
        model_name=model_name,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        tokenizer=tokenizer,
        device=device,
        num_runs=1,
        learning_rate=3e-5,
        num_epochs=5,
        batch_size=16,
        accumulation_steps=2,
        seed=42,
        auroc_weight=auroc_weight  # AUROC 가중치 전달
    )

    # Final evaluation with best model
    print("\n===== Final Evaluation with Best Overall Model =====")
    final_model = model_class.from_pretrained(model_name, num_labels=2)
    final_model.load_state_dict(best_model_state)
    final_model.to(device)

    test_dataloader = data_preparing(test_data, tokenizer, batch_size=16, shuffle=False)
    _, test_acc, test_f1, test_precision, test_recall, test_auc, test_tpr,_ = eval_model(final_model, test_dataloader, device)


start_time = time.time()

if __name__ == "__main__":
    main()

elapsed_time_sec = time.time() - start_time
elapsed_time_min = elapsed_time_sec / 60
print(f"소요 시간: {elapsed_time_min:.2f}분")

