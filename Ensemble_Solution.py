import pandas as pd
import numpy as np
import scipy
import random
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoConfig
CFG1 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp1/deberta-v3-large',
    'max_len': 64, 
    'epochs': 5,
    'train_bs': 16, 
    'valid_bs': 32,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG2 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp2/deberta-v3-large',
    'max_len': 64, 
    'epochs': 5,
    'train_bs': 16, 
    'valid_bs': 32,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 0,
    'sigmoid': False
}

CFG3 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/bert-for-patents/bert-for-patents',
    'path': '../input/upppm-exp3/bert-for-patents',
    'max_len': 64, 
    'epochs': 5,
    'train_bs': 16, 
    'valid_bs': 32,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 0,
    'sigmoid': False
}

CFG4 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp4/deberta-v3-large',
    'max_len': 64, 
    'epochs': 5,
    'train_bs': 16, 
    'valid_bs': 32,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 0,
    'sigmoid': True
}

CFG10 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp10/deberta-v3-large',
    'max_len': 64, 
    'epochs': 5,
    'train_bs': 16, 
    'valid_bs': 32,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 0,
    'sigmoid': False
}

CFG11 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/bert-for-patents/bert-for-patents',
    'path': '../input/upppm-exp11/bert-for-patents',
    'max_len': 64, 
    'epochs': 5,
    'train_bs': 16, 
    'valid_bs': 32,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 0,
    'sigmoid': False
}

CFG16 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp16/deberta-v3-large',
    'max_len': 384, 
    'epochs': 5,
    'train_bs': 8, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 0,
    'sigmoid': False
}

CFG17 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp17/deberta-v3-large',
    'max_len': 256, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG18 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-base/deberta-v3-base',
    'path': '../input/upppm-exp18/deberta-v3-base',
    'max_len': 384, 
    'epochs': 5,
    'train_bs': 8, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 0,
    'sigmoid': False
}

CFG19 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-base/deberta-v3-base',
    'path': '../input/upppm-exp19/deberta-v3-base',
    'max_len': 384, 
    'epochs': 5,
    'train_bs': 8, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 0,
    'sigmoid': False
}

CFG20 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/bert-for-patents/bert-for-patents',
    'path': '../input/upppm-exp20/bert-for-patents',
    'max_len': 384, 
    'epochs': 5,
    'train_bs': 16, 
    'valid_bs': 32,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 0,
    'sigmoid': False
}

CFG21 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppmexp21/deberta-v3-large',
    'max_len': 384, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG23 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/debertalarge',
    'path': '../input/upppm-exp23/deberta-large',
    'max_len': 500, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG26 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp26/deberta-v3-large',
    'max_len': 512, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG27 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/bert-for-patents/bert-for-patents',
    'path': '../input/upppm-exp27/bert-for-patents',
    'max_len': 512, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG28 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp28/deberta-v3-large',
    'max_len': 512, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG29 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp29/deberta-v3-large',
    'max_len': 384, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG31 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp31/deberta-v3-large',
    'max_len': 384, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': True
}

CFG32 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-xlarge',
    'path': '../input/upppm-exp32-output/upppm-exp32/deberta-xlarge',
    'max_len': 450, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG33 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp33/deberta-v3-large',
    'max_len': 512, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG35 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp35/exp35/deberta-v3-large',
    'max_len': 64, 
    'epochs': 5,
    'train_bs': 16, 
    'valid_bs': 32,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 0,
    'sigmoid': False
}

CFG36 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/bert-for-patents/bert-for-patents',
    'path': '../input/upppm-exp36/exp36/bert-for-patents',
    'max_len': 64, 
    'epochs': 5,
    'train_bs': 16, 
    'valid_bs': 32,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 0,
    'sigmoid': False
}

CFG38 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp38/exp38/deberta-v3-large',
    'max_len': 384, 
    'epochs': 5,
    'train_bs': 8, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 0,
    'sigmoid': False
}

CFG39 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/bert-for-patents/bert-for-patents',
    'path': '../input/upppm-exp39/exp39/bert-for-patents',
    'max_len': 384, 
    'epochs': 5,
    'train_bs': 16, 
    'valid_bs': 32,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 0,
    'sigmoid': False
}

CFG40 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/debertalarge',
    'path': '../input/upppm-exp40/exp40/deberta-large',
    'max_len': 450, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG49 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp49-output/upppm-exp49/deberta-v3-large',
    'max_len': 560, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG51 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/bert-for-patents/bert-for-patents',
    'path': '../input/upppm-exp51-output/upppm-exp51/bert-for-patents',
    'max_len': 512, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG61 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp61-output/upppm-exp61/deberta-v3-large',
    'max_len': 250, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG62 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/bert-for-patents/bert-for-patents',
    'path': '../input/upppm-exp62-output/upppm-exp62/bert-for-patents',
    'max_len': 250, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG63 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/debertalarge',
    'path': '../input/upppm-exp63-output/upppm-exp63/deberta-large',
    'max_len': 280, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG64 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp64-output/upppm-exp64/deberta-v3-large',
    'max_len': 320, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG69 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp69-output/upppm-exp69/deberta-v3-large',
    'max_len': 250, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG70 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/bert-for-patents/bert-for-patents',
    'path': '../input/upppm-exp70-output/upppm-exp70/bert-for-patents',
    'max_len': 320, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG71 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp71-output/upppm-exp71/deberta-v3-large',
    'max_len': 250, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG74 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp74-output/upppm-exp74/deberta-v3-large',
    'max_len': 280, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}

CFG80 = {
    'fold_num': 5,
    'seed': 42,
    'model': '../input/deberta-v3-large/deberta-v3-large',
    'path': '../input/upppm-exp80-output/upppm-exp80/deberta-v3-large',
    'max_len': 280, 
    'epochs': 5,
    'train_bs': 4, 
    'valid_bs': 16,
    'lr': 1e-5, 
    'num_workers': 2,
    'weight_decay': 1e-2,
    'sigmoid': False
}
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG1['seed'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_cpc_texts():
    contexts = []
    pattern = '[A-Z]\d+'
    for file_name in os.listdir('../input/cpc-data/CPCSchemeXML202105'):
        result = re.findall(pattern, file_name)
        if result:
            contexts.append(result)
    contexts = sorted(set(sum(contexts, [])))
    results = {}
    for cpc in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']:
        with open(f'../input/cpc-data/CPCTitleList202202/cpc-section-{cpc}_20220201.txt') as f:
            s = f.read()
        pattern = f'{cpc}\t\t.+'
        result = re.findall(pattern, s)
        cpc_result = result[0].lstrip(pattern)
        for context in [c for c in contexts if c[0] == cpc]:
            pattern = f'{context}\t\t.+'
            result = re.findall(pattern, s)
            results[context] = cpc_result + ". " + result[0].lstrip(pattern)
    return results


cpc_texts = get_cpc_texts()

train_df = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/train.csv')
test_df = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/test.csv')

train_df['flag'] = 0
test_df['flag'] = 1
test_df['score'] = -1

all_df = pd.concat([test_df, train_df], 0)

all_df['context_text'] = all_df['context'].map(cpc_texts).apply(lambda x:x.lower())
all_df = all_df.join(all_df.groupby('anchor').target.agg(list).rename('ref'), on='anchor')
all_df['ref2'] = all_df.apply(lambda x:[i for i in x['ref'] if i != x['target']], axis=1)
all_df['ref2'] = all_df.ref2.apply(lambda x: ', '.join(sorted(list(set(x)), key=x.index)))
all_df['ref'] = all_df.ref.apply(lambda x:', '.join(sorted(list(set(x)), key=x.index)))

all_df = all_df.join(all_df.groupby(['anchor', 'context']).target.agg(list).rename('ref3'), on=['anchor', 'context'])
all_df['ref3'] = all_df.apply(lambda x: ', '.join([i for i in x['ref3'] if i != x['target']]), axis=1)

all_df = all_df.join(all_df.groupby('context').anchor.agg('unique').rename('anchor_list'), on='context')
all_df['anchor_list'] = all_df.apply(lambda x:', '.join([i for i in x['anchor_list'] if i != x['anchor']]), axis=1)

all_df['text1'] = all_df['anchor'] + '[SEP]' + all_df['target'] + '[SEP]'  + all_df['context_text']
all_df['text2'] = all_df['anchor'] + '[SEP]' + all_df['target'] + '[SEP]'  + all_df['context_text'] + '[SEP]'  + all_df['ref']
all_df['text3'] = all_df['anchor'] + '[SEP]' + all_df['target'] + '[SEP]'  + all_df['context_text'] + '[SEP]'  + all_df['ref2']
all_df['text4'] = all_df['anchor'] + '[SEP]' + all_df['target'] + '[SEP]'  + all_df['context_text'] + '[SEP]'  + all_df['ref2'] + ', ' + all_df['anchor_list']
all_df['text5'] = all_df['anchor'] + '[SEP]' + all_df['target'] + '[SEP]'  + all_df['context_text'] + '[SEP]'  + all_df['ref3']
all_df['text6'] = 'The similarity between anchor ' + all_df['anchor'] + ' and target ' + all_df['target'] + '. Context is ' + all_df['context_text'] + \
            '. Candidates are ' + all_df['ref3']
all_df

class MyDataset(Dataset):
    def __init__(self, dataframe, add_ref=0):
        self.df = dataframe
        self.add_ref = add_ref
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.add_ref == 0:
            text = str(self.df.text1.values[idx])
        elif self.add_ref == 1:
            text = str(self.df.text2.values[idx])
        elif self.add_ref == 2:
            text = str(self.df.text3.values[idx])
        elif self.add_ref == 3:
            text = str(self.df.text4.values[idx])
        elif self.add_ref == 4:
            text = str(self.df.text5.values[idx])
        else:
            text = str(self.df.text6.values[idx])
        return text
MyDataset(all_df, 4)[0]

def collate_fn(data):
    text = tokenizer(data, padding='max_length', truncation=True, max_length=CFG['max_len'], return_tensors='pt')
    input_ids = text['input_ids']
    attention_mask = text['attention_mask']
    return input_ids, attention_mask

def collate_fn_fast(data):
    text = tokenizer(data, padding='longest', truncation=True, max_length=CFG['max_len'], return_tensors='pt')
    input_ids = text['input_ids']
    attention_mask = text['attention_mask']
    return input_ids, attention_mask
class Model(nn.Module):
    def __init__(self, CFG):
        super(Model, self).__init__()
        cfg = AutoConfig.from_pretrained(CFG['model'])
        cfg.num_labels=1
        self.bert = AutoModelForSequenceClassification.from_config(cfg)
 
    def forward(self, input_ids, attention_mask):
        y = self.bert(input_ids=input_ids, attention_mask=attention_mask).logits
        return y
def test_model(model, val_loader, sigmoid=False):
    model.eval()
    y_pred = []
    
    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for step, batch in enumerate(tk):
            input_ids, attention_mask = [x.to(device) for x in batch]
            
            output = model(input_ids, attention_mask).squeeze(-1)
            
            if sigmoid:
                output = output.sigmoid()
            
            y_pred.extend(output.cpu().numpy())

    return np.array(y_pred)
test_df = all_df[all_df.flag==1]

w = [0.01, 0.02, -0.03, 0.01, 0.05, 0.03, 0.05, 0.06, 0.12, 0.1, 0.07, -0.02, 0.1, 0.04, 0.03, 0.07, 0.1, 0.11, 0.05]

ensemble_predictions = []

for CFG in [CFG3, CFG4, CFG11, CFG17, CFG20, CFG32, CFG35, CFG36, CFG38, CFG39, CFG40, CFG49, CFG61, CFG62, CFG63, CFG71, CFG74, CFG80, CFG69]:
    predictions = []
    
    tokenizer = AutoTokenizer.from_pretrained(CFG['model'])

    ## sort df
    input_lengths = []
    if CFG in [CFG17, CFG19, CFG20, CFG29, CFG32]:
        for text in test_df['text2'].values:
            length = len(tokenizer(text, add_special_tokens=True)['input_ids'])
            if length > CFG['max_len']:
                length = CFG['max_len'] 
            input_lengths.append(length)
    elif CFG in [CFG38, CFG39, CFG40, CFG49]:
        for text in test_df['text3'].values:
            length = len(tokenizer(text, add_special_tokens=True)['input_ids'])
            if length > CFG['max_len']:
                length = CFG['max_len'] 
            input_lengths.append(length)
    elif CFG in [CFG61, CFG62, CFG63, CFG69, CFG71]:
        for text in test_df['text5'].values:
            length = len(tokenizer(text, add_special_tokens=True)['input_ids'])
            if length > CFG['max_len']:
                length = CFG['max_len'] 
            input_lengths.append(length)   
    elif CFG in [CFG74, CFG80]:
        for text in test_df['text6'].values:
            length = len(tokenizer(text, add_special_tokens=True)['input_ids'])
            if length > CFG['max_len']:
                length = CFG['max_len'] 
            input_lengths.append(length)     
    else:
        for text in test_df['text1'].values:
            length = len(tokenizer(text, add_special_tokens=True)['input_ids'])
            if length > CFG['max_len']:
                length = CFG['max_len'] 
            input_lengths.append(length)
    test_df['input_lengths'] = input_lengths
    length_sorted_idx = np.argsort([-len_ for len_ in input_lengths])
    sort_df = test_df.iloc[length_sorted_idx]
    

    if CFG in [CFG17, CFG19, CFG20, CFG29, CFG32]:
        test_set = MyDataset(sort_df, 1)
    elif CFG in [CFG38, CFG39, CFG40, CFG49]:
        test_set = MyDataset(sort_df, 2)
    elif CFG in [CFG61, CFG62, CFG63, CFG69, CFG71]:
        test_set = MyDataset(sort_df, 4)
    elif CFG in [CFG74, CFG80]:
        test_set = MyDataset(sort_df, 5)
    else:
        test_set = MyDataset(sort_df, 0)
    
    test_loader = DataLoader(test_set, batch_size=CFG['valid_bs'], shuffle=False, collate_fn=collate_fn_fast, num_workers=CFG['num_workers'])

    model = Model(CFG).to(device)

    for fold in range(CFG['fold_num']):
        model.load_state_dict(torch.load('{}_fold_{}.pt'.format(CFG['path'], fold)))
        prediction = test_model(model, test_loader, CFG['sigmoid'])
        
        prediction = prediction[np.argsort(length_sorted_idx)]
            
        predictions.append(prediction)
        
    predictions = np.mean(predictions, 0)
    ensemble_predictions.append(predictions)

stage1_predictions = np.sum([w[i]*ensemble_predictions[i] for i in range(len(w))], 0)

all_df.loc[all_df.flag==0, 'oof'] = np.load('../input/upppm-oof/oof_cv8697.npy')
all_df.loc[all_df.flag==1, 'oof'] = ((stage1_predictions-stage1_predictions.min())/(stage1_predictions.max()-stage1_predictions.min())*100).round().astype('int')

all_df['target_oof'] = all_df['target'] + ' ' + all_df['oof'].astype('str')
all_df = all_df.join(all_df.groupby('anchor').target_oof.agg(list).rename('ref4'), on='anchor')
all_df.ref4 = all_df.ref4.apply(lambda x:sorted(list(set(x)), key=x.index))

all_df = all_df.join(all_df.groupby(['anchor', 'context']).target_oof.agg(list).rename('ref5'), on=['anchor', 'context'])

test_df = all_df[all_df.flag==1]

test_df['ref4'] = test_df.apply(lambda x:', '.join([i for i in x['ref4'] if i != x['target_oof']]), axis=1)
test_df['ref5'] = test_df.apply(lambda x:', '.join([i for i in x['ref5'] if i != x['target_oof']]), axis=1)
test_df['text2'] = test_df['anchor'] + '[SEP]' + test_df['target'] + '[SEP]' + test_df['context_text'] + '[SEP]' + test_df['ref4']
test_df['text4'] = test_df['anchor'] + '[SEP]' + test_df['target'] + '[SEP]' + test_df['context_text'] + '[SEP]' + test_df['ref5']
w = [0.01, 0.02, -0.03, 0.01, 0.04, 0.03, 0.04, 0.04, 0.08, 0.06, 0.05, -0.03, 0.07, -0.01, 0.02, 0.04, 0.09, 0.09, 0.05, 0.01, 0.09, 0.12]

for CFG in [CFG26, CFG64, CFG70]:
    predictions = []
    
    tokenizer = AutoTokenizer.from_pretrained(CFG['model'])

    ## sort df
    input_lengths = []
    if CFG in [CFG26, CFG33]:
        for text in test_df['text2'].values:
            length = len(tokenizer(text, add_special_tokens=True)['input_ids'])
            if length > CFG['max_len']:
                length = CFG['max_len'] 
            input_lengths.append(length)
    else:
        for text in test_df['text4'].values:
            length = len(tokenizer(text, add_special_tokens=True)['input_ids'])
            if length > CFG['max_len']:
                length = CFG['max_len'] 
            input_lengths.append(length)
    test_df['input_lengths'] = input_lengths
    length_sorted_idx = np.argsort([-len_ for len_ in input_lengths])
    sort_df = test_df.iloc[length_sorted_idx]
    
    if CFG in [CFG26, CFG33]:
        test_set = MyDataset(sort_df, 1)
    else:
        test_set = MyDataset(sort_df, 4)
    test_loader = DataLoader(test_set, batch_size=CFG['valid_bs'], shuffle=False, collate_fn=collate_fn_fast, num_workers=CFG['num_workers'])

    model = Model(CFG).to(device)

    for fold in range(CFG['fold_num']):
        model.load_state_dict(torch.load('{}_fold_{}.pt'.format(CFG['path'], fold)))
        prediction = test_model(model, test_loader, CFG['sigmoid'])
        
        prediction = prediction[np.argsort(length_sorted_idx)]
            
        predictions.append(prediction)
        
    predictions = np.mean(predictions, 0)
    ensemble_predictions.append(predictions)

final_predictions = np.sum([w[i]*ensemble_predictions[i] for i in range(len(w))], 0)


submission = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/sample_submission.csv')
submission['score'] = final_predictions
submission[['id', 'score']].to_csv('submission.csv', index=False)

# https://www.kaggle.com/code/zzy990106/upppm-ensemble-two-stage-final
