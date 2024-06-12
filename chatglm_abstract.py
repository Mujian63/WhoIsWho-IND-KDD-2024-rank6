import pandas as pd
import numpy as np

import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold,GroupKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
import gc
import re
from sklearn.metrics import roc_auc_score
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import polars as pl
from pathlib import Path
from glob import glob
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
import joblib

import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold,GroupKFold,KFold
from tqdm import tqdm
import gc
import re
from sklearn.metrics import roc_auc_score
import os

from datasets import concatenate_datasets,load_dataset,load_from_disk
from transformers import AutoModel, AutoTokenizer, AdamW, DataCollatorWithPadding

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pytl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import polars as pl

import sys
 
# 参数索引从0开始，sys.argv[0]是脚本名，后面是传递的参数

iii = int(sys.argv[1])
print(iii)

train = pd.read_feather('data/train.feather')
valid = pd.read_feather('data/valid.feather')
test = pd.read_feather('data/test.feather')

piddf = joblib.load('data/pid_df.pkl')

data = pd.concat([train,valid,test]).reset_index(drop = True)
piddf = piddf[piddf['id'].isin(set(data['PID']))]

piddf['abstract'] = piddf['abstract'].apply(lambda x:x.lower()[:1024])

import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value
        
        
model = AutoModel.from_pretrained("THUDM/chatglm3-6b",trust_remote_code=True)
chatglm3_tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm3-6b',trust_remote_code=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

intermediate_output = None

def hook_fn(module, input, output):
    global intermediate_output
    intermediate_output = output
    
def get_chatglm3_text2vec(text):
    global intermediate_output
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    intermediate_output = None
    
    model.transformer.encoder.register_forward_hook(hook_fn)

    # 输入文本
    inputs = chatglm3_tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)

    # 进行前向传播
    with torch.no_grad():
        outputs = model(**inputs)
    return np.mean(intermediate_output[0].to('cpu').numpy(),0)[0]




t_dict = dict()

start_idx = iii*3000
ans = []
for pid,text in tqdm(piddf[['id','abstract']][start_idx:start_idx+3000].values):
    t_dict[pid] = get_chatglm3_text2vec(text)
    


print(iii*3000,iii*3000+3000)
    
joblib.dump(t_dict,f'abstract/t_dict_abstract_{iii}.pkl')