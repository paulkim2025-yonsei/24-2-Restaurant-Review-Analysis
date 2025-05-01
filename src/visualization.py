from flask import Flask, request, render_template
import pandas as pd
import plotly.express as px

app = Flask(__name__)

summary_1 = pd.read_excel('../dataset/summary_광고비처리.xlsx')
summary_2 = pd.read_excel('../dataset/summary_찐리뷰만.xlsx')

columns_list = [
    "유자유김치떡볶이_신촌점",
    "크리스터_치킨",
    "빠빠빠치킨_연대본점",
    "에일크루브루잉_신촌점",
    "목구멍_신촌점",
    "하나마토_신촌점",
    "착한곱창",
    "오향미엔",
    "홍미닭발",
    "김덕후의곱창조_신촌점",
    "이자카야_우규_신촌점",
    "고삼이_신촌점",
    "신촌정직한족발",
    "더도이축산직영점_신촌점"
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_weights = {
        '맛있음': int(request.form.get('맛있음', 0)),
        '위생': int(request.form.get('위생', 0)),
        '서비스': int(request.form.get('서비스', 0)),
        '분위기': int(request.form.get('분위기', 0)),
        '위치접근성': int(request.form.get('위치접근성', 0)),
        '대기시간': int(request.form.get('대기시간', 0)),
        '가성비': int(request.form.get('가성비', 0)),
        '가격': int(request.form.get('가격', 0))
    }
    n_top = int(request.form.get('top_n', 5))
    score_threshold = float(request.form.get('score_threshold', 0))
    sort_by = request.form.get('sort_by', '평균 점수')

    def calculate_scores(summary_df):
        summary_df['추천 점수'] = summary_df.apply(
            lambda row: row['긍정도 (%)'] * user_weights.get(row['클래스 설명'], 0)
            if row['클래스 설명'] in user_weights else 0,
            axis=1
        )
        final_scores = summary_df.groupby('식당 이름')['추천 점수'].sum().sort_values(ascending=False)

        def rename_func(x):
            try:
                idx = int(x.split('_')[1]) - 1
                if 0 <= idx < len(columns_list):
                    return columns_list[idx]
                else:
                    return x
            except:
                return x

        final_scores_renamed = final_scores.rename(rename_func)
        return final_scores_renamed

    scores_1 = calculate_scores(summary_1)
    scores_2 = calculate_scores(summary_2)

    result_1 = scores_1.to_frame(name='광고성 리뷰 포함').reset_index()
    result_1.rename(columns={'index': '식당 이름'}, inplace=True)
    result_2 = scores_2.to_frame(name='광고성 리뷰 제거').reset_index()
    result_2.rename(columns={'index': '식당 이름'}, inplace=True)

    merged_results = pd.merge(result_1, result_2, on='식당 이름', how='outer').fillna(0)
    merged_results['평균 점수'] = merged_results[['광고성 리뷰 포함', '광고성 리뷰 제거']].mean(axis=1)
    merged_results = merged_results[merged_results['평균 점수'] >= score_threshold]
    merged_results = merged_results.sort_values(by=sort_by, ascending=False).head(n_top)

    min_value = merged_results[['광고성 리뷰 포함','광고성 리뷰 제거']].values.min() * 0.8
    fig = px.bar(
        merged_results.melt(id_vars='식당 이름', var_name='데이터셋', value_name='추천 점수'),
        x='추천 점수',
        y='식당 이름',
        color='데이터셋',
        barmode='group',
        title="신촌 식당 추천 결과",
        text='추천 점수'
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis=dict(range=[min_value, merged_results[['광고성 리뷰 포함','광고성 리뷰 제거']].values.max() * 1.2]),
        annotations=[
            dict(
                text="제작자: 산업공학회 PIE 24-2 DS 학회원 (강태희, 고민지, 김건우, 김세원, 김채연, 안성진)",
                xref="paper", yref="paper",
                x=0, y=-0.2,
                showarrow=False
            )
        ]
    )

    graph_html = fig.to_html(full_html=False)
    return render_template('result.html', graph_html=graph_html)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
