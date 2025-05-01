import torch
import fasttext
import numpy as np
import re
import json
import pandas as pd
from konlpy.tag import Okt
import torch.nn as nn

class LSTMAttentionModel(nn.Module):
    def __init__(self, i=300, h=32, sl=800):
        super(LSTMAttentionModel, self).__init__()
        self.sl = sl
        self.h = h
        self.lstm = nn.LSTM(i, h, batch_first=True)
        self.weight = nn.Linear(sl, sl)
        self.fc = nn.Linear(h*sl, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out, _ = self.lstm(x)
        p = out.permute(0, 2, 1)
        q = self.weight(p)
        q = q.permute(0, 2, 1)
        qk = out * q
        return self.fc(self.flatten(qk)), qk

def output_to_prob(logit):
    return torch.sigmoid(logit)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '../dataset/model.pth'
model = LSTMAttentionModel()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

ft = fasttext.load_model('../dataset/cc.ko.300.bin')
with open('../dataset/stopwords.json', 'r', encoding='utf-8') as f_s:
    stopwords = json.load(f_s)

okt = Okt()

def preprocess_text(text):
    text = re.sub(r'#\S+', '', str(text)).strip()
    tokens = okt.morphs(text)
    filtered = [w for w in tokens if w not in stopwords and w.isalpha()]
    vectors = [ft.get_word_vector(word) for word in filtered]
    if len(vectors) < 800:
        vectors += [np.zeros(300)] * (800 - len(vectors))
    else:
        vectors = vectors[:800]
    return np.array(vectors)

def predict_advertisement(text, threshold=0.9):
    embs = preprocess_text(text)
    embs = torch.tensor(embs).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output, _ = model(embs)
        prob = output_to_prob(output)
        label = 1 if prob.item() > threshold else 0
    return label

if __name__ == "__main__":
    data1 = pd.read_csv('../dataset/data1.csv')
    filtered_reviews = []
    for i, row in data1.iterrows():
        if 'review' in data1.columns:
            review_text = row['review']
        else:
            review_text = ''

        if len(str(review_text)) < 10:
            continue
        is_adv = predict_advertisement(review_text)
        if is_adv == 0:
            filtered_reviews.append(review_text)

    print("광고성 리뷰 제외 후 남은 리뷰 수:", len(filtered_reviews))

    final_df = pd.DataFrame({'review': filtered_reviews})
    final_df.to_csv('../dataset/data1_filtered.csv', index=False, encoding='utf-8-sig')
