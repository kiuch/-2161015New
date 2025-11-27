import pandas as pd
import MeCab
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import os
# 複数のレビューファイルの設定
file_config = [
    {'title': 'SV', 'path': r'C:\Users\masat\OneDrive\デスクトップ\deep learning\パワポ\-2161015New\SVレビュー文.csv', 'review_col': 'レビュー'},
     {'title': '剣盾', 'path': r'C:\Users\masat\OneDrive\デスクトップ\deep learning\パワポ\-2161015New\剣盾シナリオ文.csv', 'review_col': 'シナリオ一文'},
    {'title': 'USUM', 'path': r'C:\Users\masat\OneDrive\デスクトップ\deep learning\パワポ\-2161015New\sm_usumシナリオ文.csv', 'review_col': 'シナリオ文'}
]

# ストップワード (汎用的な単語や特定のゲーム用語を除去)
stop_words = {
    "この", "の", "は", "が", "に", "を", "と", "て", "た", "だ", "し", "もっと", "も", "です", "ます", "けど", "だろ", "それ", 
    "いう", "ある", "もの", "なる", "する", "いる", "こと", "ない", "できる", "ため", "そノ", "られる", "れる", "これ", 
    "スル", "イル", "イウ", "アル", "ナル", "ナイ", "コト", "デキル", "シレル", "カンズル", "モノ",
    "ゲーム", "シリーズ", "ポケモン", "ホンサク", "ルート", "ブブン", 
    "レベル", "タメ", "ソノ", "セイリツ", "トオク", "ミエル", "ハツ", "イク", "クル", "オク",
    "ホカク", "シュルイ", "タチバ", "マチ", "イチ", "アタリ", "バアイ", "ジム", "要素", "システム", 
    "感想", "点", "部分", "今回","感じ", "思った", "ところ", "また" 
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
        
        # N-gram=1の単語リストを生成
        if hinshi in target_hinshi and original_form_for_check not in stop_words and len(surface_form) > 1:
            words.append(surface_form)
        
        node = node.next
    return words

def plot_horizontal_bar_charts(df_list, title_list, filename):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8)) 
    
    for i, (df, title) in enumerate(zip(df_list, title_list)):
        ax = axes[i]
        words = df["word"].values[::-1]
        counts = df["wordcount"].values[::-1]
        
        ax.barh(words, counts, color='#4682B4')
        ax.set_title(f'{title} の単語頻出度', fontsize=14)
        ax.set_xlabel('出現回数')
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=10)
        
    fig.suptitle(f"作品別 単語頻出度比較 (N-gram = 1)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(filename)
    plt.close()
    print(f"✅ 3作品比較の棒グラフを '{filename}' として保存しました。")

def main():
    print("作品別 単語頻出度比較分析を開始します...")

    if not os.path.exists('results'):
        os.makedirs('results')

    df_for_plot = []
    titles_for_plot = []
    
    # --- 作品ごとの分析ループ ---
    for config in file_config:
        title = config['title']
        path = config['path']
        review_col = config['review_col']
        
        print(f"\n==================== 📊 {title} の処理を開始 ====================")
        
        df = force_read_csv(path)
        df_game = df.copy()
        df_game = df_game.rename(columns={review_col: 'Original_Review'})
        df_game['Original_Review'] = df_game['Original_Review'].astype(str).str.strip().replace('nan', '')
        df_game = df_game[df_game['Original_Review'].str.len() > 1]
        game_reviews = df_game['Original_Review'].tolist()
        
        # 形態素解析と単語リスト生成
        processed_reviews = [preprocess_text(review, mecab) for review in game_reviews]
        
        freq_dict = defaultdict(int)
        for review_words in processed_reviews:
            for word in review_words: # N-gram=1 (単語)
                freq_dict[word] += 1
                
        # 頻度順にソートしたDataFrameを生成
        fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
        fd_sorted.columns = ["word", "wordcount"]
        
        df_for_plot.append(fd_sorted.head(20)) 
        titles_for_plot.append(title)
        
        print(f"✅ {title} の単語頻出度 (上位10単語):")
        for index, row in fd_sorted.head(10).iterrows():
            print(f"  {row['word']}: {row['wordcount']}回")
        
        print(f"==================== ✅ {title} の処理を完了 ====================")
    if len(df_for_plot) == 3:
        plot_horizontal_bar_charts(
            df_for_plot, 
            titles_for_plot, 
            'results/word_frequency_comparison_bar_chart.png'
        )
print("\n--- 全処理を完了しました ---")

if __name__ == "__main__":
    main()