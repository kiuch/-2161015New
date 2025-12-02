import pandas as pd
import MeCab
from collections import Counter
import matplotlib.pyplot as plt
import os
import numpy as np


# 複数のレビューファイルの設定 (ユーザー指定の絶対パスを含む)
file_config = [
    {'title': 'SV', 'path': r'C:\Users\masat\OneDrive\デスクトップ\deep learning\パワポ\-2161015New\SVレビュー文.csv', 'review_col': 'レビュー'},
    {'title': '剣盾', 'path': r'C:\Users\masat\OneDrive\デスクトップ\deep learning\パワポ\-2161015New\剣盾シナリオ文.csv', 'review_col': 'シナリオ一文'},
    {'title': 'USUM', 'path': r'C:\Users\masat\OneDrive\デスクトップ\deep learning\パワポ\-2161015New\sm_usumシナリオ文.csv', 'review_col': 'シナリオ文'}
]

# ストップワード (形態素解析後のフィルタリングに使用)
stop_words = {
    "この", "の", "は", "が", "に", "を", "と", "て", "た", "だ", "し", "もっと", "も", "です", "ます", "けど", "だろ", "それ", 
    "いう", "ある", "もの", "なる", "する", "いる", "こと", "できる", "ため", "そノ", "られる", "れる", "これ", 
    "スル", "イル", "イウ", "アル", "ナル", "ナイ", "コト", "デキル", "シレル", "カンズル", "モノ",
    "ゲーム", "シリーズ", "ポケモン",  "ホンサク", "ルート", "ブブン", 
    "レベル", "タメ", "ソノ", "セイリツ", "トオク", "ミエル", "ハツ", "イク", "クル", "オク",
    "ホカク", "シュルイ", "マチ", "イチ", "アタリ", "バアイ", "ジム", "テラスタル", "要素", "システム", 
    "感想", "点", "部分", "今回",  "感じ", "思った", "ところ", "また",
    "キャラ" 
}
positive_words = {
    "スバラシイ", "カンドウ", "サイコウ", "メイサク", "オモシロイ", "ヨイ", "スキ", "コエル","テイネイ",
    "セットクリョク", "ボツニュウカン", "タカイ", "ナク", "カミ", "タノシイ", 
    "キタイ", "アツイ", "カワイイ", "ツナガル", "シュウイツ", "シンセン", "リアル",
    "キワダツ", "カンセイド", "マッチ", "ネッチュウ", "ヒキコマレル", "ケッサク", 
    "ミゴト", "ワクワク", "ボリューム", "アイチャク", "イトシイ", "ナットク", 
    "シッカリ", "マンゾク", "セイチョウ", "キフク", "ミロクテキ", "タノシム",
    "サイコウホウ", "シュウバン", "ナケル", "キタイイジョウ","ハッピー",
}
negative_words = {
    "ヨワイ", "ヘイボン", "ザンネン", "チンプ", "サイアク", "ストレス", "ナイ","アンマリ","ヒクイ",
    "ヒョウカデキナイ", "ビミョウ", "アッサリ", "コドモムケ", "ツマラナイ", 
    "フマン", "ワルイ", "オクレ", "クソ", "モンダイ", "ソガイ", "メンドウ", 
    "コンナン", "ツタナサ", "モノタリナイ", "キタイハズレ", "タンチョウ", 
    "フカイカン", "イミフメイ", "ウスイ", "タイクツ", "チセツ", "シリツボミ", 
    "カッテ", "フカンゼン", "アサイ", "セッキョウクサイ", "サイテイ","アキル","ウスッペライ","モノタリナイ","デキナイ",
}

try:
    mecab = MeCab.Tagger() 
except Exception as e:
    print(f" MeCabの初期化に失敗しました: {e}")
    exit()

plt.rcParams['font.family'] = 'Meiryo' 
plt.rcParams['font.size'] = 12


def force_read_csv(file_path):
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

def unify_words(word):
    if word == 'キャラ':
        return 'キャラクター'
    return word

def preprocess_text(text, mecab_tagger):
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
        
        # 感情分析の際は「基本形（原形）」の取得を試みる
        original_form = node.surface 
        if len(features) >= 7 and features[6] != '*':
            # 7番目のフィールドが基本形（原形）
            original_form = features[6] 
            print(f"Surface: {node.surface}, Hinshi: {hinshi}, BasicForm: {original_form}")

        if hinshi in target_hinshi and original_form not in stop_words:
            # 抽出する単語は基本形とする
            processed_word = unify_words(original_form) 
            words.append(processed_word)

        node = node.next
    return words

def analyze_sentiment(words):
    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)
    
    if positive_score > negative_score:
        sentiment = 'Positive'
    elif negative_score > positive_score:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
        
    return sentiment, positive_score, negative_score

def plot_sentiment_distribution(df_data, file_name, title): 
    """感情極性の分布を円グラフで可視化する (色の対応を固定)"""
    fixed_order = ['Positive', 'Negative', 'Neutral'] 
    colors = ['#66b3ff', '#ff9999', '#99ff99'] # 青=Positive, 赤=Negative, 緑=Neutral
    
    sentiment_counts = df_data['Sentiment'].value_counts().reindex(fixed_order, fill_value=0) 
    
    if not sentiment_counts.empty:
        plt.figure(figsize=(6, 6))
        plt.pie(
            sentiment_counts, 
            labels=sentiment_counts.index, 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=colors 
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()

def main():
    print("作品別 感情分析を開始します...")

    if not os.path.exists('results'):
        os.makedirs('results')

    all_analyzed_dfs = []

    # --- 作品ごとの分析ループ ---
    for config in file_config:
        title = config['title']
        path = config['path']
        review_col = config['review_col']
        
        print(f"\n==================== 📈 {title} の処理を開始 ====================")
        
        df = force_read_csv(path)
        if df is None or review_col not in df.columns:
            print(f"エラー: {title}のファイル読み込みまたは列名'{review_col}'の確認に失敗しました。スキップします。")
            continue

        df_game = df.copy()
        df_game['Game_Title'] = title 
        df_game = df_game.rename(columns={review_col: 'Original_Review'})
        df_game['Original_Review'] = df_game['Original_Review'].astype(str).str.strip().replace('nan', '')
        df_game = df_game[df_game['Original_Review'].str.len() > 1].reset_index(drop=True)
        game_reviews = df_game['Original_Review'].tolist()
        
        processed_reviews = [preprocess_text(review, mecab) for review in game_reviews]
        
        # 感情分析の実行
        sentiment_results = [analyze_sentiment(words) for words in processed_reviews]
        sentiment_df = pd.DataFrame(sentiment_results, columns=['Sentiment', 'Positive_Score', 'Negative_Score'])

        df_game['Sentiment'] = sentiment_df['Sentiment']
        df_game['Positive_Score'] = sentiment_df['Positive_Score']
        df_game['Negative_Score'] = sentiment_df['Negative_Score']
        
        # 感情極性の分布を可視化
        filename = f'results/{title}_sentiment_distribution_pie_chart.png'
        plot_sentiment_distribution(df_game, filename, title=f'{title} レビュー感情極性の分布')
        print(f"✅ 感情分析結果を円グラフ '{filename}' として保存しました。")

        # 結果をCSVに保存
        output_path = f'results/{title}_sentiment_analysis_results.csv'
        df_game.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ 詳細結果を '{output_path}' に保存しました。")

    print("\n--- 感情分析スクリプトの全処理を完了しました ---")

if __name__ == "__main__":
    main()