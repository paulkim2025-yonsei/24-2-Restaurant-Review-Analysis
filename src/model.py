import fasttext
import pandas as pd
import numpy as np
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

ft = fasttext.load_model('../dataset/cc.ko.300.bin')

with open('../dataset/stopwords.json', 'r', encoding='utf-8') as f_s:
    stopwords = json.load(f_s)

okt = Okt()

adv_yn_data1 = pd.read_csv('../dataset/data_연남.csv')
adv_yn_data2 = pd.read_csv('../dataset/data_신촌.csv')
adv = pd.concat([adv_yn_data1, adv_yn_data2], ignore_index=True)

text = adv['text']
label = adv['label']
print("전체 데이터 수:", len(text), "광고성(1) 수:", sum(label))

def preprocess_and_embed(texts):
    embs = []
    for t in texts:
        t = re.sub(r'#\S+', '', str(t)).strip()
        tokens = okt.morphs(t)
        filtered = [w for w in tokens if w not in stopwords and w.isalpha()]
        vectors = [ft.get_word_vector(word) for word in filtered]
        if len(vectors) < 800:
            vectors += [np.zeros(300)] * (800 - len(vectors))
        else:
            vectors = vectors[:800]
        embs.append(vectors)
    return np.array(embs)

emb = preprocess_and_embed(text)
print("임베딩 완료 shape:", emb.shape)

X_train, X_test, y_train, y_test = train_test_split(
    emb, label, test_size=0.2, stratify=label, random_state=0
)

class AdvDataset(Dataset):
    def __init__(self, emb, label):
        self.emb = torch.tensor(emb).float()
        self.y = torch.tensor(label.values).float()
    def __getitem__(self, idx):
        return self.emb[idx], self.y[idx]
    def __len__(self):
        return len(self.y)

train_ds = AdvDataset(X_train, y_train)
test_ds = AdvDataset(X_test, y_test)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

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

def output_to_label(logit):
    prob = torch.sigmoid(logit)
    return (prob > 0.5).squeeze().int()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LSTMAttentionModel()
model.to(device)
criterion = nn.BCEWithLogitsLoss()
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 1000
best_test_loss = float('inf')
patience = 0

for epoch in tqdm(range(epochs)):
    model.train()
    train_loss = 0.0
    train_correct = 0
    for x_batch, y_batch in train_dl:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output, _ = model(x_batch)
        loss = criterion(output.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (output_to_label(output) == y_batch).sum().item()

    train_loss /= len(train_ds)
    train_acc = train_correct / len(train_ds)

    model.eval()
    test_loss = 0.0
    test_correct = 0
    with torch.no_grad():
        for x_batch, y_batch in test_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output, _ = model(x_batch)
            loss = criterion(output.squeeze(), y_batch)
            test_loss += loss.item()
            test_correct += (output_to_label(output) == y_batch).sum().item()
    test_loss /= len(test_ds)
    test_acc = test_correct / len(test_ds)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model = model
        patience = 0
    else:
        patience += 1
        if patience >= 100:
            lr *= 0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            patience = 0

    if epoch % 50 == 0:
        print(
            f"[Epoch {epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}"
        )
    if test_loss <= 0.0001:
        break

best_model.eval()
all_emb_tensor = torch.tensor(emb).float().to(device)
with torch.no_grad():
    out, attn = best_model(all_emb_tensor)
pred_labels = output_to_label(out).cpu().numpy()
true_labels = label.values
print(classification_report(pred_labels, true_labels))
f1 = f1_score(true_labels, pred_labels)
print("F1 Score:", f1)

torch.save(best_model.state_dict(), '../dataset/model.pth')
print("학습된 모델 가중치 저장 완료")

_, feat = best_model(all_emb_tensor)
feat_np = feat.cpu().detach().numpy()
np.save('../dataset/ft', feat_np)
print("ft.npy 저장 완료")
