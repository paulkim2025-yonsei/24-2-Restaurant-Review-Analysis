# 신촌 식당 추천 시스템

**발표자료**: [DS 최종발표.pdf](DS최종발표.pdf)

## 1. 프로젝트 개요
- **목표**  
  네이버 블로그 리뷰 중 “광고성 리뷰”를 분류·제거하고,  
  식당 특성(맛있음·위생·서비스·분위기·위치접근성·대기시간·가성비·가격)별 긍정도와  
  사용자가 선택한 중요도(weight)를 종합하여 최적의 식당을 추천합니다.

- **주요 단계**  
  1. **크롤링** (`crowling.py`)  
  2. **데이터 분석** (`clustering.ipynb`, `pca.ipynb`, `topic.ipynb`)  
  3. **광고성 리뷰 분류 모델 학습** (`model.py`)  
  4. **광고성 리뷰 필터링** (`adv_filtering.py`)  
  5. **웹 시각화(Flask)** (`visualization.py`, `templates/index.html`, `templates/result.html`)

---

## 2. 주요 구성 파일

### `crowling.py`
- **기능**: Selenium으로 네이버 지도 리뷰를 무한 스크롤·“더보기” 클릭하여 전수 수집  
- **입력**: `hashtag_lists`에 식당명·식별자 목록  
- **출력**: `식당명_리뷰.xlsx`

### `clustering.ipynb`
- **기능**: `pca.npy` 데이터를 K-Means 클러스터링 → Calinski–Harabasz, Silhouette 지수 비교  
- **출력**: `clustering.csv`

### `pca.ipynb`
- **기능**: `ft.npy`(LSTM-Attention 특성 벡터) → MinMaxScaler → PCA  
- **출력**: `pca.npy`

### `model.py`
- **기능**:  
  1. FastText(`cc.ko.300.bin`) + `stopwords.json` 로드  
  2. `data_연남.csv`, `data_신촌.csv` 합쳐서 광고성(1)/비광고성(0) 학습  
  3. LSTM-Attention 모델 학습 → `model.pth`, `ft.npy` 저장

### `adv_filtering.py`
- **기능**:  
  1. `model.pth` 로드 → 광고성 확률 > 0.9 리뷰 제거  
  2. 원본 데이터(`data1.csv` 등)에서 필터링 → `data1_filtered.csv`

### `visualization.py`
- **기능**: Flask 웹 서버  
  - `summary_광고비처리.xlsx` vs `summary_찐리뷰만.xlsx` 비교  
  - 사용자 입력(맛있음·위생·서비스·…·가격) 기반 추천 점수 계산  
  - Plotly 바 차트 시각화 → `templates/result.html`

### `templates/index.html`
- 사용자에게 기준별 중요도, 최소 점수 기준, 상위 N개, 정렬 기준 입력 폼 제공

### `templates/result.html`
- 서버에서 계산된 그래프 출력 및 “다시 입력하기” 링크

---

## 3. 모델링 (사용한 기법/이론)

- **Word Embedding**  
  - FastText 기반 `cc.ko.300.bin` 사용: subword 단위 학습으로 OOV 문제 완화  
  - 리뷰 전처리: 정규표현식→형태소 분석(Okt)→불용어(`stopwords.json`+날짜·시간 리스트) → 최대 길이 800 토큰 패딩  

- **LSTM-Attention 분류 모델**  
  - Attention: Attention Map을 지속적으로 확인하여 모델이 올바르게 돌아가고 있는지 수시로 확인하고 필요한 경우 불용어 처리 진행함.  
  - 구조:  
    self.lstm = nn.LSTM(input_dim=300, hidden_dim=32, batch_first=True)  
    self.weight = nn.Linear(seq_len, seq_len)  
    self.fc     = nn.Linear(hidden_dim * seq_len, 1)  
  - 핵심 키워드 오분류 보완을 위해 `CheckAttention`으로 비핵심 단어 3,171개 수동 불용어 처리  

- **차원 축소 & 클러스터링**  
  1. t-SNE, PCA로 2D·3D 차원 축소 후 시각화  
  2. 최적 클러스터 수 탐색:  
     - Calinski–Harabasz 지수 → 군집 내/간 분산 비율  
     - Silhouette 지수 → 군집 응집도 vs 분리도  
     - 최종 8개 클래스 선택 (맛있음·위생·서비스·분위기·위치·대기시간·가성비·가격)  

- **감성 분석**  
  - 클래스별 사전 정의된 핵심 키워드 기반 긍정/부정 판별  
  - 코사인 유사도 ≥ 1/3 임계치 적용  

- **추천 점수 계산**  
  ranked_weights = {c: w * (len(rank)-i) for i,c in enumerate(rank)}  
  summary_df['추천 점수'] = summary_df.apply(  
      lambda r: r['긍정도(%)'] * ranked_weights.get(r['클래스'],0),  
      axis=1  
  )

---

## 4. 결과

- **광고성 리뷰 분류 성능**  
  - F1 Score: **0.9811**, Accuracy: **0.9685**  

- **필터링 효과**  
  - 기준 확률 0.9 적용 시 전체 리뷰의 **20–30%**가 광고성으로 제거  

- **클래스별 긍정도 Example**  
  - Restaurant 1의 “맛있음” 클래스: 총 리뷰 25개 중 긍정 13, 부정 12 → 긍정도 52%  

- **추천 시스템 비교**  
  - 광고성 리뷰 **포함 vs 제거** 결과 비교(탐복·구도로통닭)  
    - ‘탐복’: 제거 후 점수가 더 높아 **진짜 맛집**  
    - ‘구도로통닭’: 포함 시 점수 급등 → 광고성 리뷰가 많이 포함된 **인스타 바이럴 맛집** 가능성  

- **웹 서비스 배포**  
  - 웹 서비스 배포 또한 진행함. (웹 링크는 제공하지 않음)

---

## 팀 및 기여

- **팀명**: 연세대학교 산업공학과 학회 PIE DS 24-2 전공 스터디  
- **구성원**: 강태희, 고민지, 김건우(팀장), 김세원, 김채연, 안성진


---
---

# Shinchon Restaurant Recommendation System

**Presentation Slides**: [DS Final Presentation.pdf](DS최종발표.pdf)

---

## 1. Project Overview

- **Objective**  
  Classify and remove “promotional reviews” from Naver Blog review data, (Naver is Korean version of Google!) 
  then compute sentiment scores for each restaurant attribute (Deliciousness·Hygiene·Service·Atmosphere·Accessibility·Waiting Time·Cost-Effectiveness·Price) and combine them with user-selected weights to recommend the best restaurant.

- **Main Steps**  
  1. **Crawling** (`crowling.py`)  
  2. **Data Analysis** (`clustering.ipynb`, `pca.ipynb`, `topic.ipynb`)  
  3. **Promotional Review Classification Model Training** (`model.py`)  
  4. **Promotional Review Filtering** (`adv_filtering.py`)  
  5. **Web Visualization (Flask)** (`visualization.py`, `templates/index.html`, `templates/result.html`)

---

## 2. Key Files

### `crowling.py`
- **Function**: Uses Selenium to infinite-scroll and click “More” on Naver Map reviews to collect all entries.  
- **Input**: `hashtag_lists` containing pairs of restaurant names and identifiers.  
- **Output**: `식당명_리뷰.xlsx`

### `clustering.ipynb`
- **Function**: Performs K-Means clustering on `pca.npy` features; compares Calinski–Harabasz and Silhouette scores.  
- **Output**: `clustering.csv`

### `pca.ipynb`
- **Function**: Loads `ft.npy` (LSTM-Attention feature vectors), applies MinMaxScaler, then PCA.  
- **Output**: `pca.npy`

### `model.py`
- **Function**:  
  1. Loads FastText (`cc.ko.300.bin`) and `stopwords.json`.  
  2. Merges `data_연남.csv` and `data_신촌.csv` for promotional (1) vs. non-promotional (0) labeling.  
  3. Trains an LSTM-Attention classifier → saves `model.pth` and `ft.npy`.

### `adv_filtering.py`
- **Function**:  
  1. Loads `model.pth` → removes any review with promotional probability > 0.9.  
  2. Filters the original datasets (`data1.csv`, etc.) → outputs `data1_filtered.csv`.

### `visualization.py`
- **Function**: Flask web server that:  
  - Compares `summary_광고비처리.xlsx` vs. `summary_찐리뷰만.xlsx`.  
  - Calculates recommendation scores based on user-entered weights (Deliciousness·Hygiene·Service·Atmosphere·Accessibility·Waiting Time·Cost-Effectiveness·Price).  
  - Renders Plotly bar charts in `templates/result.html`.

### `templates/index.html`
- Presents a form for users to select attribute weights, minimum score threshold, top-N results, and sorting preference.

### `templates/result.html`
- Displays the computed charts and a “Try Again” link.

---

## 3. Modeling (Techniques & Theory)

- **Word Embedding**  
  - FastText (`cc.ko.300.bin`): subword-level training to mitigate OOV issues.  
  - Preprocessing: regex → morphological analysis (Okt) → remove stopwords (`stopwords.json` + date/time list) → pad to max length of 800 tokens.

- **LSTM-Attention Classification Model**  
  - **Attention Monitoring**: Continuously inspect attention maps; refine stopword list as needed.  
  - **Architecture**:  
    self.lstm   = nn.LSTM(input_dim=300, hidden_dim=32, batch_first=True)  
    self.weight = nn.Linear(seq_len, seq_len)  
    self.fc     = nn.Linear(hidden_dim * seq_len, 1)  
  - **Manual Stopword Refinement**: Used `CheckAttention` to identify 3,171 non-core tokens and add them to the stopword list to correct misclassifications.

- **Dimensionality Reduction & Clustering**  
  1. Applied t-SNE and PCA for 2D/3D visualization.  
  2. Determined optimal number of clusters using the Calinski–Harabasz index (inter- vs. intra-cluster variance) and Silhouette score (cohesion vs. separation).  
  3. Selected **8 clusters** corresponding to the classes: 맛있음 (deliciousness), 위생 (hygiene), 서비스 (service), 분위기 (atmosphere), 위치접근성 (accessibility), 대기시간 (waiting time), 가성비 (cost-effectiveness), 가격 (price).

- **Sentiment Analysis**  
  - For each class, used a predefined set of core keywords.  
  - Applied a cosine similarity threshold of ≥ 1/3 to decide positive vs. negative sentiment.

- **Recommendation Score Calculation**  
  ranked_weights = {c: w * (len(rank) - i) for i, c in enumerate(rank)}  
  summary_df['recommendation_score'] = summary_df.apply(  
      lambda r: r['positive_ratio(%)'] * ranked_weights.get(r['class'], 0),  
      axis=1  
  )

---

## 4. Results

- **Promotional Review Classification Performance**  
  - F1 Score: **0.9811**  
  - Accuracy: **0.9685**

- **Filtering Impact**  
  - At a 0.9 probability threshold, **20–30%** of all reviews were removed as promotional.

- **Class-wise Sentiment Example**  
  - For Restaurant 1’s “맛있음” class: out of 25 reviews, 13 were positive and 12 negative → **52% positive**.

- **Recommendation System Comparison**  
  - **With vs. Without Promotional Reviews**  
    - **탐복**: score increased after filtering → truly popular spot  
    - **구도로통닭**: score spiked before filtering → likely driven by viral/promotion-heavy content

- **Web Service Deployment**  
  - A live web service was deployed (link not publicly shared).

---

## Team & Contributions

- **Team**: PIE DS 24-2 Major Study, Industrial Engineering Society, Yonsei University  
- **Members**: 강태희, 고민지, 김건우(KIM GEONWOO, Team Leader), 김세원, 김채연, 안성진
