# -*- coding: utf-8 -*-

geimport sys
sys.version

# 한글 깨짐 방지
# !sudo apt-get install -y fonts-nanum
# !sudo fc-cache -fv
# !rm ~/.cache/matplotlib -rf

# 설치
# !pip install konlpy
!pip install transformers torch

from huggingface_hub import login

# Hugging Face API 토큰 입력
login(token="")

# from huggingface_hub import whoami
# print(whoami())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.metrics import roc_auc_score

# 회귀용
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# 머신러닝 알고리즘 - 분류
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

# 머신러닝 알고리즘 - 회귀
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor

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

# 워드 클라우드를 위한 라이브러리
from collections import Counter
from wordcloud import WordCloud
from IPython.display import Image

# 저장
import pickle

# torch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler
from transformers import get_scheduler

from google.colab import drive
drive.mount('/content/drive')

"""preprocessing(1)

### 전처리
"""



# 데이터 합치기

# 파일 경로 설정
file_paths = [ '찬성반대_combined_250317'
,'VL_주장_combined'
,'TL_주장_combined'
,'VL_설명글_combined'
,'TL_설명글_combined'
,'VL_대안제시_combined'
,'TL_대안제시_combined'
,'VL_글짓기_combined'
,'TL_글짓기_combined']

data_dict = {}

for i, f in enumerate(file_paths, start=1):
    file_path = f'/content/drive/My Drive/25.1H Thesis/{f}.xlsx'
    data = pd.read_excel(file_path)
    data_dict[f'data_{i}'] = data

for key, df in data_dict.items():
    print(f"{key} 첫 번째 행:")
    print(df.iloc[0])
    print()

combined_data = pd.concat(data_dict.values(), ignore_index=True)

print("병합된 데이터프레임:")
print(combined_data.head())

# CSV 저장
combined_data.to_csv('/content/drive/My Drive/25.1H Thesis/combined_data.csv', index=False, encoding='utf-8')
print("CSV 파일 저장 완료: combined_data.csv")

data_c = combined_data.explode('paragraph', ignore_index=True)

print(data_c['paragraph'].isnull().sum())
print(data_c['paragraph'].head())
print(data_c['paragraph'].apply(type).value_counts())

# paragraph 모으기
def safe_eval(val):
    if isinstance(val, str):  # 값이 문자열일 경우
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return None
    return val


data_c['paragraph'] = data_c['paragraph'].apply(safe_eval)

print(data_c['paragraph'])

data_c['paragraph_txt'] = data_c['paragraph'].apply(lambda x: x[0].get('paragraph_txt') if isinstance(x, list) and len(x) > 0 else None)

data_c['combined_paragraph_txt'] = data_c['paragraph'].apply(
    lambda x: ' '.join([item['paragraph_txt'] for item in x]) if isinstance(x, list) else '')

print(data_c['combined_paragraph_txt'].iloc[0])
print(data_c[['paragraph', 'combined_paragraph_txt']].head())

# 불필요한 단어, 공백 제거

data_c['combined_paragraph_txt'] = data_c['combined_paragraph_txt'].str.replace('#@문장구분#', '', regex=False)
data_c['combined_paragraph_txt'] = data_c['combined_paragraph_txt'].str.replace('\n\n', '', regex=False)
print(data_c['combined_paragraph_txt'].iloc[0])

data_c['info.essay_prompt'] = data_c['info.essay_prompt'].str.replace('\n\n', '', regex=False)
print(data_c['info.essay_prompt'].iloc[0])

# human writing 0, machine writing 1

data_c['target'] = '0'
print(data_c.iloc[0])

data_c.to_csv('/content/drive/MyDrive/25.1H Thesis/combined_data_preprocessed.csv', index=False)

file_path = '/content/drive/MyDrive/25.1H Thesis/combined_data_preprocessed.csv'
data1 = pd.read_csv(file_path)

data1.info()

## machine writing

# 파일 경로 설정
file_path_g0 = '/content/drive/My Drive/25.1H Thesis/machine_GPT4o_250408.csv'
file_path_g1 = '/content/drive/My Drive/25.1H Thesis/machine_GPT3_5_250409.csv'
file_path_g2 = '/content/drive/My Drive/25.1H Thesis/machine_GPT3_5_250410.csv'
file_path_g3 = '/content/drive/My Drive/25.1H Thesis/machine_GPT4_0_250412.csv'
file_path_g4 = '/content/drive/My Drive/25.1H Thesis/machine_GPT4_0_250415.csv'
file_path_g5 = '/content/drive/My Drive/25.1H Thesis/machine_GPT4_0_250417.csv'


file_path_b0 = '/content/drive/My Drive/25.1H Thesis/machine_gemini2_0_250408.csv'
file_path_b1 = '/content/drive/My Drive/25.1H Thesis/machine_gemini1_5_250409.csv'
file_path_b2 = '/content/drive/My Drive/25.1H Thesis/machine_gemini2_0_250410.csv'
file_path_b3 = '/content/drive/My Drive/25.1H Thesis/machine_gemini2_0_250412.csv'
file_path_b4 = '/content/drive/My Drive/25.1H Thesis/machine_gemini1_5_250415.csv'
file_path_b5 = '/content/drive/My Drive/25.1H Thesis/machine_gemini2_0_250415.csv'
file_path_b6 = '/content/drive/My Drive/25.1H Thesis/machine_gemini2_0_250417.csv'
file_path_b7 = '/content/drive/My Drive/25.1H Thesis/machine_gemini2_0_250418.csv'
file_path_b8 = '/content/drive/My Drive/25.1H Thesis/machine_gemini1_5_250419.csv'
file_path_b9 = '/content/drive/My Drive/25.1H Thesis/machine_gemini2_0_250419.csv'
file_path_b10 = '/content/drive/My Drive/25.1H Thesis/machine_gemini2_0_250420.csv'



data0 = pd.read_csv(file_path_g0)
data0["model"] = "gpt-4o"

data1 = pd.read_csv(file_path_g1)
data1["model"] = "gpt-3.5-turbo"

data2 = pd.read_csv(file_path_g2)
data2["model"] = "gpt-3.5-turbo"

data3 = pd.read_csv(file_path_g3)
data3["model"] = "gpt-4o"

data4 = pd.read_csv(file_path_g4)
data4["model"] = "gpt-4o"

data5 = pd.read_csv(file_path_g5)
data5["model"] = "gpt-4o"

data6 = pd.read_csv(file_path_b0)
data6["model"] = "gemini-2.0-flash"

data7 = pd.read_csv(file_path_b1)
data7["model"] = "gemini-1.5-flash"

data8 = pd.read_csv(file_path_b2)
data8["model"] = "gemini-2.0-flash"

data9 = pd.read_csv(file_path_b3)
data9["model"] = "gemini-2.0-flash"

data10 = pd.read_csv(file_path_b4)
data10["model"] = "gemini-1.5-flash"

data11 = pd.read_csv(file_path_b5)
data11["model"] = "gemini-2.0-flash"

data12 = pd.read_csv(file_path_b6)
data12["model"] = "gemini-2.0-flash"

data13 = pd.read_csv(file_path_b7)
data13["model"] = "gemini-2.0-flash"

data14 = pd.read_csv(file_path_b8)
data14["model"] = "gemini-1.5-flash"

data15 = pd.read_csv(file_path_b9)
data15["model"] = "gemini-2.0-flash"

data16 = pd.read_csv(file_path_b10)
data16["model"] = "gemini-2.0-flash"

# 데이터프레임 결합
machine = pd.concat([data0, data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15, data16], ignore_index=True)

machine.info()

print(machine.groupby('model')['machine_generated_text'].count())

# 불필요한 공백 제거

machine['info.essay_prompt'] = machine['info.essay_prompt'].str.replace('\n\n', '', regex=False)
print(machine['info.essay_prompt'].iloc[11])

machine['machine_generated_text'] = machine['machine_generated_text'].str.replace('\n\n', '', regex=False)

machine['target'] = '1'
machine.info()

machine['machine_generated_text'].iloc[10]

machine.to_csv('/content/drive/MyDrive/25.1H Thesis/combined_machine_preprocessed_250419.csv', index=False)

"""preprocessing(2)"""

## human writing
data1 = pd.read_csv('/content/drive/MyDrive/25.1H Thesis/combined_data_preprocessed.csv')

## machine writing
data2 = pd.read_csv('/content/drive/MyDrive/25.1H Thesis/combined_machine_preprocessed_250419.csv',)

# 데이터 확인
print(data1.info(), data2.info())

# 1. 텍스트 길이

data1['model'] = 'human'
rs_1 = data1[['combined_paragraph_txt', 'info.essay_prompt', 'student.student_grade', 'info.essay_main_subject','model', 'target']]
rs_1.rename(columns = {'combined_paragraph_txt' : 'txt'}, inplace = True)

print(rs_1.iloc[0])
rs_1['txt_len'] = rs_1['txt'].str.len()
rs_1.info()

rs_2 = data2[['machine_generated_text', 'info.essay_prompt', 'student.student_grade', 'info.essay_main_subject', 'model','target']]
rs_2.rename(columns = {'machine_generated_text' : 'txt'}, inplace = True)
rs_2['txt_len'] = rs_2['txt'].str.len()

rs_2.info()

rs = pd.concat([rs_1, rs_2],ignore_index=True)
print(rs[rs['target'] == 1]['txt'])

## 형태소 분석
okt = Okt()

# 유일한 단어 개수 계산 (명사, 동사, 형용사만)
def count_unique_words_with_konlpy(text):
    pos_tags = okt.pos(text)  # 형태소 및 품사 태깅
    valid_pos = {'Noun', 'Adjective', 'Verb'}  # 추출 대상 품사
    filtered_tokens = [word for word, pos in pos_tags if pos in valid_pos]
    return len(set(filtered_tokens))  # 중복 제거 후 단어 수 계산

# 각 txt 행마다 고유 단어 수 계산
tqdm.pandas()
# rs['unique_word_count'] = rs['txt'].apply(count_unique_words_with_konlpy)
rs_2['unique_word_count'] = rs_2['txt'].progress_apply(count_unique_words_with_konlpy)

# rs.to_csv('/content/drive/My Drive/25.1H Thesis/combined_data_250412_1.csv', index=False, encoding='utf-8')
rs_2.to_csv('/content/drive/My Drive/25.1H Thesis/m_combined_data_250419_1.csv', index=False, encoding='utf-8')

# machine : 중3~고3
grades = ['고등_3학년','고등_2학년','고등_1학년','중등_3학년']

prac1 = pd.read_csv('/content/drive/My Drive/25.1H Thesis/m_combined_data_250419_1.csv')
rs_temp1 = prac1[(prac1['target']==1) & (prac1['student.student_grade'].isin(grades))]

rs_temp1.info()

print(rs_temp1.groupby('model')['txt'].count())

rs_temp1.to_csv('/content/drive/My Drive/25.1H Thesis/m_combined_data_250419_2.csv', index=False, encoding='utf-8')

# human : 중3~고3
prac2 = pd.read_csv('/content/drive/My Drive/25.1H Thesis/combined_data_250412_1.csv')

grades = ['고등_3학년','고등_2학년','고등_1학년','중등_3학년']

rs_temp2 = prac2[(prac2['target']==0) & (prac2['student.student_grade'].isin(grades))]

rs_temp2.info()

# Commented out IPython magic to ensure Python compatibility.
# # 2. 맞춤법 오류건수 : 크롤링
# 
# # Set up for running selenium in Google Colab
# ## You don't need to run this code if you do it in Jupyter notebook, or other local Python setting
# 
# %%shell
# sudo apt -y update
# sudo apt install -y wget curl unzip
# wget http://archive.ubuntu.com/ubuntu/pool/main/libu/libu2f-host/libu2f-udev_1.1.4-1_all.deb
# dpkg -i libu2f-udev_1.1.4-1_all.deb
# wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
# dpkg -i google-chrome-stable_current_amd64.deb
# CHROME_DRIVER_VERSION=`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`
# wget -N https://chromedriver.storage.googleapis.com/$CHROME_DRIVER_VERSION/chromedriver_linux64.zip -P /tmp/
# unzip -o /tmp/chromedriver_linux64.zip -d /tmp/
# chmod +x /tmp/chromedriver
# mv /tmp/chromedriver /usr/local/bin/chromedriver
# pip install selenium

!pip install chromedriver-autoinstaller

import sys
sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')

from bs4 import BeautifulSoup
from selenium import webdriver
import chromedriver_autoinstaller

# setup chrome options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless') # ensure GUI is off
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# set path to chromedriver as per your configuration
chromedriver_autoinstaller.install()

# set up the webdriver
driver = webdriver.Chrome(options=chrome_options)

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import sys
from selenium.webdriver.common.keys import Keys
import urllib.request
import os
from urllib.request import urlretrieve
import chromedriver_autoinstaller  # setup chrome options

chrome_path = "/content/drive/MyDrive/Colab Notebooks/chromedriver"

sys.path.insert(0,chrome_path)
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless') # ensure GUI is off : cloab은 새창을 지원하지않기 때문에 창 없는 모드
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')  # set path to chromedriver as per your configuration
chrome_options.add_argument('lang=ko_KR') # 한국어

chromedriver_autoinstaller.install()  # set the target URL

driver.get("https://www.google.com")
print(driver.title)  # "Google" should be printed
driver.quit()

sys.path.insert(0,chrome_path)
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless') # ensure GUI is off : cloab은 새창을 지원하지않기 때문에 창 없는 모드
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')  # set path to chromedriver as per your configuration
chrome_options.add_argument('lang=ko_KR') # 한국어
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36")

driver = webdriver.Chrome(options=chrome_options)
driver.get("https://www.saramin.co.kr/zf_user/tools/character-counter")
print(driver.title)

print(driver.page_source)

def check_spelling(text, driver):
    try:
        driver.get("https://www.saramin.co.kr/zf_user/tools/character-counter")
        wait = WebDriverWait(driver, 10)

        # 페이지가 완전히 로드될 때까지 대기
        wait.until(lambda driver: driver.execute_script('return document.readyState') == 'complete')

        # 입력창 찾기
        input_box = wait.until(EC.presence_of_element_located((By.ID, "character_counter_content")))
        input_box.clear()
        input_box.send_keys(text)

        # 맞춤법 검사 버튼 클릭
        submit_button = wait.until(EC.element_to_be_clickable((By.ID, "spell_check")))
        submit_button.click()

        # 맞춤법 검사 결과 대기 (최대 10초)
        spell_count_element = wait.until(EC.visibility_of_element_located((By.ID, "spell_count")))

        # 전체 텍스트 출력 (디버깅)
        spell_count_text = spell_count_element.text.strip()

        # 숫자만 추출하는 정규식 적용
        match = re.search(r'\d+', spell_count_text)
        spell_count_value = match.group() if match else "0"

        return int(spell_count_value)

    except Exception as e:
        print(f"오류 발생: {e}")
        return None

# 사용 예시
text = '머라고 하는건지 이해가안되요'
corrected_text = check_spelling(text, driver)

print(corrected_text)

# machine
tqdm.pandas(desc="맞춤법 검사 진행 중")

batch_size = 500
num_batches = (len(rs_temp1) + batch_size - 1) // batch_size  # 총 배치 수

# 맞춤법 검사 적용
for i in tqdm(range(num_batches), desc="배치 진행 중"):
    batch = rs_temp1.iloc[i * batch_size:(i + 1) * batch_size].copy()
    batch['spelling_result'] = batch['txt'].progress_apply(lambda x: check_spelling(x, driver))

    # 파일 저장
    filename = f'/content/drive/MyDrive/25.1H Thesis/modeling_mstxt_250419_batch_{i+1}.csv'
    batch.to_csv(filename, index=False)
    print(f"Batch {i+1} 저장 완료: {filename}")

# WebDriver 종료
driver.quit()

# human
tqdm.pandas(desc="맞춤법 검사 진행 중")

batch_size = 500
num_batches = (len(rs_temp2) + batch_size - 1) // batch_size  # 총 배치 수

# 맞춤법 검사 적용
for i in tqdm(range(20,num_batches), desc="배치 진행 중"):
    batch = rs_temp2.iloc[i * batch_size:(i + 1) * batch_size].copy()
    batch['spelling_result'] = batch['txt'].progress_apply(lambda x: check_spelling(x, driver))

    # 파일 저장
    filename = f'/content/drive/MyDrive/25.1H Thesis/modeling_hstxt_250420_batch_{i+1}.csv'
    batch.to_csv(filename, index=False)
    print(f"Batch {i+1} 저장 완료: {filename}")

# WebDriver 종료
driver.quit()

datah = []
datam = []

# 첫 번째 데이터 리스트 로드
for i in range(32):
    filename = f'/content/drive/MyDrive/25.1H Thesis/modeling_hstxt_250420_batch_{i+1}.csv'
    datah.append(pd.read_csv(filename))

# 두 번째 데이터 리스트 로드
for i in range(21):
    filename = f'/content/drive/MyDrive/25.1H Thesis/modeling_mstxt_250419_batch_{i+1}.csv'
    datam.append(pd.read_csv(filename))

# 모든 데이터 병합
rs = pd.concat(datah + datam, ignore_index=True)

print(rs.groupby('target')['txt'].count())

# 이상치 제거
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 이상치 제거
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

rs_temp3 = remove_outliers_iqr(rs, 'txt_len')

rs_temp3.info()

print(rs_temp3.groupby('target')['txt'].count(), rs_temp3.groupby('target')['txt'].count()/rs_temp3['txt'].count())

# 3. 글자수 대비 유일한 단어 비중
rs_temp3['unique_word_ratio'] = (rs_temp3['unique_word_count'] / rs_temp3['txt_len']).round(2)

# 결과 확인
print(rs_temp3[rs_temp3['target']==1].head())

print(rs_temp3.groupby('target')['unique_word_ratio'].mean())

# 결측치 처리
print(rs_temp3['spelling_result'].isna().sum())

rs_temp3 = rs_temp3[rs_temp3['spelling_result'].isna() == False]

# 3. 글자수 대비 맞춤법 오류 비중
rs_temp3['spelling_ratio'] = (rs_temp3['spelling_result'] / rs_temp3['txt_len']).round(2)

# 결과 확인
print(rs_temp3[rs_temp3['target']==1].head())

print(rs_temp3.groupby('target')['spelling_ratio'].mean())

# 난이도점수 평가 : human/machine 텍스트 분류의 어려움 -> minmax scaler로 정규화

scaler = MinMaxScaler()

# 정규화 대상 컬럼
columns_to_scale = ['unique_word_ratio','txt_len','spelling_ratio']

# 정규화 적용
rs_sample_scaled = rs_temp3.copy()
rs_sample_scaled[columns_to_scale] = scaler.fit_transform(rs_temp3[columns_to_scale])

# difficulty_score 계산 (0~1 범위 내)

rs_temp3['difficulty_score1'] = rs_sample_scaled['txt_len']
rs_temp3['difficulty_score2'] = rs_sample_scaled['unique_word_ratio']
rs_temp3['difficulty_score3'] = rs_sample_scaled['spelling_ratio']

rs_temp3.to_csv('/content/drive/My Drive/25.1H Thesis/combined_data_250423f.csv', index=False, encoding='utf-8')

"""여기부터"""

rs_sample = pd.read_csv('/content/drive/MyDrive/25.1H Thesis/combined_data_250423f.csv')
rs_sample.info()

# print(rs_sample.groupby(['student.student_grade','target'])['txt'].count())
print(rs_sample.groupby(['info.essay_main_subject', 'target'])['txt'].count())

plt.figure(figsize=(4, 3))
sns.scatterplot(data=rs_sample, x='target', y='txt_len', hue='target')
plt.title('Text Length vs Target')

plt.show()

# 생략
# sampling (프롬프트별 360개 중복없이)

unique_prompts = rs_sample['info.essay_main_subject'].unique()
grades = ['고등_3학년','고등_2학년','고등_1학년','중등_3학년']

random_sample1 = rs_sample[(rs_sample['info.essay_main_subject'].isin(unique_prompts)) & (rs_sample['target'] == 0)& (rs_sample['student.student_grade'].isin(grades))].groupby('info.essay_main_subject', group_keys=False).apply(lambda x: x.sample(n=360, random_state=42, replace=False))
random_sample2 = rs_sample[(rs_sample['info.essay_main_subject'].isin(unique_prompts)) & (rs_sample['target'] == 1)& (rs_sample['student.student_grade'].isin(grades))].groupby('info.essay_main_subject', group_keys=False).apply(lambda x: x.sample(n=360, random_state=42, replace=False))

print(random_sample1.info(), random_sample2.info())

# 실험대상 데이터 병합
rs_sample = pd.concat([random_sample1, random_sample2], ignore_index=True)

rs_sample.to_csv('/content/drive/My Drive/25.1H Thesis/sampled_data_250423f.csv', index=False, encoding='utf-8')
