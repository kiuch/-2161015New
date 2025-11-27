import pandas as pd
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
from collections import Counter 

# 複数のレビューファイルの設定
file_config = [
    {'title': 'SV', 'path': r'C:\Users\masat\OneDrive\デスクトップ\deep learning\パワポ\-2161015New\SVレビュー文.csv', 'review_col': 'レビュー'},
    {'title': '剣盾', 'path': r'C:\Users\masat\OneDrive\デスクトップ\deep learning\パワポ\-2161015New\剣盾シナリオ文.csv', 'review_col': 'シナリオ一文'},
    {'title': 'USUM', 'path': r'C:\Users\masat\OneDrive\デスクトップ\deep learning\パワポ\-2161015New\sm_usumシナリオ文.csv', 'review_col': 'シナリオ文'}
]

# ストップワード (形態素解析のフィルタリングに使用)
stop_words = {
    "この", "の", "は", "が", "に", "を", "と", "て", "た", "だ", "し", "もっと", "も", "です", "ます", "けど", "だろ", "それ", 
    "いう", "ある", "もの", "なる", "する", "いる", "こと", "ない", "できる", "ため", "そノ", "られる", "れる", "これ", 
    "スル", "イル", "イウ", "アル", "ナル", "ナイ", "コト", "デキル", "シレル", "カンズル", "モノ",
    "ゲーム", "シリーズ", "ポケモン", "ホンサク", "ルート", "ブブン", 
    "レベル", "タメ", "ソノ", "セイリツ", "トオク", "ミエル", "ハツ", "イク", "クル", "オク", 
    "ホカク", "シュルイ","マチ", "イチ", "アタリ", "バアイ", "ジム", "要素", "システム", 
    "感想", "点", "部分", "今回", "感じ", "思った", "ところ", "また" 
}

# MeCab Taggerの初期化
try:
    mecab = MeCab.Tagger() 
except Exception as e:
    print(f"MeCabの初期化に失敗しました: {e}")
    exit()

# Matplotlibの日本語設定
plt.rcParams['font.family'] = 'Meiryo' 
plt.rcParams['font.size'] = 12

# --- 2. ユーティリティ関数（データ読み込み・前処理） ---

def force_read_csv(file_path):
    """複数のエンコーディングを試してCSVを読み込む"""
    encodings_to_try = ['utf-8', 'shift_jis', 'cp932', 'euc-jp']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            return df
        except Exception:
            continue
    try:
        df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
        return df
    except Exception:
        return None

def preprocess_text(text, mecab_tagger):
    """テキストを形態素解析し、名詞・動詞・形容詞・感動詞の原形を抽出"""
    words = []
    if not isinstance(text, str) or len(text) < 2:
        return []
        
    target_hinshi = ('名詞', '動詞', '形容詞', '感動詞')
    
    try:
        node = mecab_tagger.parseToNode(text)
    except Exception:
        return []

    while node:
        features = node.feature.split(',')
        hinshi = features[0]
        
        original_form_for_check = node.surface
        if len(features) >= 7 and features[6] != '*':
            original_form_for_check = features[6]

        surface_form = node.surface
        
        # NOTE: n-gram生成を外部で行うため、ここでは単語リストを生成する
        if hinshi in target_hinshi and original_form_for_check not in stop_words and len(surface_form) > 1:
            words.append(surface_form)
        
        node = node.next
    return words

def generate_ngrams(token_list, n_gram=1):
    # TF-IDFは単語リストではなく、スペース区切りの文字列を必要とする
    token = [t for t in token_list if t != ""] 
    if not token:
        return []
        
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]


def extract_feature_words(terms, tfidfs, i, n):
    # tfidfsは密行列（toarray()後）
    tfidf_array = tfidfs[i]
    top_n_idx = tfidf_array.argsort()[-n:][::-1]
    words = [terms[idx] for idx in top_n_idx]
    scores = [tfidf_array[idx] for idx in top_n_idx]
    return list(zip(words, scores))
def main():
    print("TF-IDFを用いた作品間特徴語抽出を開始します...")

    if not os.path.exists('results'):
        os.makedirs('results')

    titles = [c['title'] for c in file_config]
    combined_reviews_by_title = {}
   
    for config in file_config:
        title = config['title']
        path = config['path']
        review_col = config['review_col']
        
        print(f"\n==================== {title} の前処理を開始 ====================")
        
        df = force_read_csv(path)
        df_game = df.copy()
        df_game = df_game.rename(columns={review_col: 'Original_Review'})
        df_game['Original_Review'] = df_game['Original_Review'].astype(str).str.strip().replace('nan', '')
        df_game = df_game[df_game['Original_Review'].str.len() > 1]
        game_reviews = df_game['Original_Review'].tolist()
        
        # 形態素解析とフィルタリング
        processed_reviews = [preprocess_text(review, mecab) for review in game_reviews]
        
        all_ngrams = []
        for review in processed_reviews:
             # TF-IDFのために、n-gram生成（単語リストをスペース区切り文字列に戻す）
            all_ngrams.extend(generate_ngrams(review, n_gram=1)) 
            
        combined_reviews_by_title[title] = " ".join(all_ngrams)
        print(f"✅ {title} のレビュー結合体を作成しました。（総単語数: {len(all_ngrams)}）")

    document_list = [combined_reviews_by_title[title] for title in titles]
    
    tfidf_vectorizer = TfidfVectorizer(
        min_df = 0.0, 
        ngram_range=(1, 2) 
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(document_list)
    terms = tfidf_vectorizer.get_feature_names_out()
    tfidfs = tfidf_matrix.toarray()

    print("\n==================== 📈 TF-IDF行列の計算完了 ====================")
    print(f"✅ 分析対象のN-gram数は {len(terms)} 種類です。")

    
    n_features = 50 # 各作品で上位50個の特徴語を抽出
    all_feature_data = []

    print(f"\n==================== 🗝️ 作品別 特徴語ランキング (上位{n_features}語) ====================")
    
    for i, title in enumerate(titles):
        feature_words_scores = extract_feature_words(terms, tfidfs, i, n_features)
        
        df_feature = pd.DataFrame(feature_words_scores, columns=['Feature_Word_Ngram', 'TFIDF_Score'])
        df_feature['Game_Title'] = title
        df_feature['Rank'] = range(1, len(df_feature) + 1)
        all_feature_data.append(df_feature)
        
        print(f"\n--- {title} の特徴語 ---")
        print(df_feature[['Rank', 'Feature_Word_Ngram', 'TFIDF_Score']].head(10))

    # 全結果を統合してCSV出力
    df_all_features = pd.concat(all_feature_data, ignore_index=True)
    output_path = 'results/tfidf_key_feature_words.csv'
    df_all_features.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n✅ 全作品の特徴語（上位{n_features}語）を '{output_path}' に保存しました。")
  
    for title in titles:
        df_plot = df_all_features[df_all_features['Game_Title'] == title].head(10)
        
        plt.figure(figsize=(10, 6))
        # TF-IDFスコアに基づいて棒グラフを作成
        plt.barh(df_plot['Feature_Word_Ngram'], df_plot['TFIDF_Score'], color='#4682B4')
        plt.title(f'{title} を最も特徴づける単語 (TF-IDF Top 10)', fontsize=14)
        plt.xlabel('TF-IDF Score')
        plt.ylabel('単語 / N-gram')
        # グラフを逆順にして、長い単語も表示可能にする
        plt.gca().invert_yaxis() 
        plt.tight_layout()
        plt.savefig(f'results/{title}_tfidf_top10_features.png')
        plt.close()
        print(f"✅ {title} のTF-IDF棒グラフを保存しました。")

    print("\n--- 全処理を完了しました ---")

if __name__ == "__main__":
    main()