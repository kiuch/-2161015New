import pandas as pd
import MeCab
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import os
import numpy as np


# 複数のレビューファイルの設定 (ユーザーが指定した絶対パスを使用)
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
    "ゲーム", "シリーズ", "ポケモン","ホンサク", "プレーヤー", "ルート", "ブブン", 
    "レベル", "タメ", "ソノ", "セイリツ", "トオク", "ミエル", "ハツ", "イク", "クル", "オク"
    "ホカク", "シュルイ", "タチバ", "マチ", "イチ", "アタリ", "バアイ", "ジム", "テラスタル", "要素", "システム", 
    "感想", "点", "部分", "今回", "感じ", "思った", "ところ", "また",
    "キャラ" 
}

# 4つの評価観点と評価語 (没入感のネガティブ評価語を強化)
evaluation_aspects = {
    "起承転結の明確性": {
        'aspect_words': ["テンカイ", "ケツマツ", "クライマックス", "ストーリー", "ナガレ", "コウセイ", "シナリオ", "フクセン"],
        'positive_eval_words': [
            "ナットク", "シッカリ", "セイゴウセイ", "ロンリテキ", "カンペキ", "フクセン", "ミゴト", "アツイ", 
            "オドロク", "ヨソウガイ", "シュウバン", "ミロクテキ", "カンジョウイニュウ", "コセイテキ", 
            "アイチャク", "サイコウ", "イキイキ", "テイネイ", "ヨイキャラ", "スキ", "キョウカン", 
            "ミヂカ", "カンドウ", "フカイ", "カンガエサセラレル", "アタタカイ", "フヘンテキ", 
            "タイセツ", "ナケル", "ボツニュウカン", "ヒキコマレル", "ワクワク", "タノシイ", 
            "ジユウド", "リアル", "フンイキ", "ココロオドル", "マンキツ"
        ],
        'negative_eval_words': [
            "シリツボミ", "ムジュン", "フカンゼン", "イミフメイ", "トウトツ", "チンプ", "ウスイ", 
            "アサイ", "チセツ", "ハタン", "ヨワイ", "ヘイボン", "コセイガ", "ミリョクガ", 
            "キョウカン", "ナカミガナイ", "メンドウ", "セッキョウクサイ", "モノタリナイ", 
            "ヒビカナイ", "カロスギ", "ミジカオスギ", "ヒョウメンジョウ", "タンチョウ", 
            "タイクツ", "ストレス", "イドウ", "サギョウ", "トオイ", "カワラズ", "バグ", 
            "カクカク", "オモイ", "マップ"
        ]
    },
    "キャラクターの魅力": {
        'aspect_words': ["キャラクター", "シュジンコウ", "ナカマ", "トウジョウジンブツ", "ライバル", "センセイ", "ジムリーダー"],
        'positive_eval_words': [
            "ナットク", "シッカリ", "セイゴウセイ", "ロンリテキ", "カンペキ", "フクセン", "ミゴト", "アツイ", 
            "オドロク", "ヨソウガイ", "シュウバン", "ミロクテキ", "カンジョウイニュウ", "コセイテキ", 
            "アイチャク", "サイコウ", "イキイキ", "テイネイ", "ヨイキャラ", "スキ", "キョウカン", 
            "ミヂカ", "カンドウ", "フカイ", "カンガエサセラレル", "アタタカイ", "フヘンテキ", 
            "タイセツ", "ナケル", "ボツニュウカン", "ヒキコマレル", "ワクワク", "タノシイ", 
            "ジユウド", "リアル", "フンイキ", "ココロオドル", "マンキツ"
        ],
        'negative_eval_words': [
            "シリツボミ", "ムジュン", "フカンゼン", "イミフメイ", "トウトツ", "チンプ", "ウスイ", 
            "アサイ", "チセツ", "ハタン", "ヨワイ", "ヘイボン", "コセイガ", "ミリョクガ", 
            "キョウカン", "ナカミガナイ", "メンドウ", "セッキョウクサイ", "モノタリナイ", 
            "ヒビカナイ", "カロスギ", "ミジカオスギ", "ヒョウメンジョウ", "タンチョウ", 
            "タイクツ", "ストレス", "イドウ", "サギョウ", "トオイ", "カワラズ", "バグ", 
            "カクカク", "オモイ", "マップ"
        ]
    },
    "テーマの身近さ": {
        'aspect_words': ["キズナ", "ユウジョウ", "テーマ", "メッセージ", "ニンゲンカンケイ", "セイチョウ", "カンジョウ"],
        'positive_eval_words': [
            "ナットク", "シッカリ", "セイゴウセイ", "ロンリテキ", "カンペキ", "フクセン", "ミゴト", "アツイ", 
            "オドロク", "ヨソウガイ", "シュウバン", "ミロクテキ", "カンジョウイニュウ", "コセイテキ", 
            "アイチャク", "サイコウ", "イキイキ", "テイネイ", "ヨイキャラ", "スキ", "キョウカン", 
            "ミヂカ", "カンドウ", "フカイ", "カンガエサセラレル", "アタタカイ", "フヘンテキ", 
            "タイセツ", "ナケル", "ボツニュウカン", "ヒキコマレル", "ワクワク", "タノシイ", 
            "ジユウド", "リアル", "フンイキ", "ココロオドル", "マンキツ"
        ],
        'negative_eval_words': [
            "シリツボミ", "ムジュン", "フカンゼン", "イミフメイ", "トウトツ", "チンプ", "ウスイ", 
            "アサイ", "チセツ", "ハタン", "ヨワイ", "ヘイボン", "コセイガ", "ミリョクガ", 
            "キョウカン", "ナカミガナイ", "メンドウ", "セッキョウクサイ", "モノタリナイ", 
            "ヒビカナイ", "カロスギ", "ミジカオスギ", "ヒョウメンジョウ", "タンチョウ", 
            "タイクツ", "ストレス", "イドウ", "サギョウ", "トオイ", "カワラズ", "バグ", 
            "カクカク", "オモイ", "マップ"
        ]
    },
    "冒険の没入感": {
        'aspect_words': ["ボウケン", "タンサク", "タビ", "フィールド", "セカイカン", "ブタイ", "タイケン", "オープンワールド"],
        'positive_eval_words': [
            "ナットク", "シッカリ", "セイゴウセイ", "ロンリテキ", "カンペキ", "フクセン", "ミゴト", "アツイ", 
            "オドロク", "ヨソウガイ", "シュウバン", "ミロクテキ", "カンジョウイニュウ", "コセイテキ", 
            "アイチャク", "サイコウ", "イキイキ", "テイネイ", "ヨイキャラ", "スキ", "キョウカン", 
            "ミヂカ", "カンドウ", "フカイ", "カンガエサセラレル", "アタタカイ", "フヘンテキ", 
            "タイセツ", "ナケル", "ボツニュウカン", "ヒキコマレル", "ワクワク", "タノシイ", 
            "ジユウド", "リアル", "フンイキ", "ココロオドル", "マンキツ"
        ],
        'negative_eval_words': [
            "シリツボミ", "ムジュン", "フカンゼン", "イミフメイ", "トウトツ", "チンプ", "ウスイ", 
            "アサイ", "チセツ", "ハタン", "ヨワイ", "ヘイボン", "コセイガ", "ミリョクガ", 
            "キョウカン", "ナカミガナイ", "メンドウ", "セッキョウクサイ", "モノタリナイ", 
            "ヒビカナイ", "カロスギ", "ミジカオスギ", "ヒョウメンジョウ", "タンチョウ", 
            "タイクツ", "ストレス", "イドウ", "サギョウ", "トオイ", "カワラズ", "バグ", 
            "カクカク", "オモイ", "マップ"
        ]
    }
}

try:
    mecab = MeCab.Tagger() 
except Exception as e:
    print(f" MeCabの初期化に失敗しました: {e}")
    exit()

plt.rcParams['font.family'] = 'Meiryo' 
plt.rcParams['font.size'] = 12


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

def unify_words(word):
    """単語を統一するルール"""
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


def calculate_co_occurrence_score(processed_words_list):
    """4つの評価観点ごとの共起分析スコアを計算する"""
    
    aspect_scores = {}
    
    for aspect_name, definitions in evaluation_aspects.items():
        aspect_score = 0
        aspect_total_count = 0
        
        for words in processed_words_list:
            word_set = set(words)
            has_aspect_word = any(w in word_set for w in definitions['aspect_words'])
            
            if has_aspect_word:
                pos_co_occurrences = sum(1 for w in definitions['positive_eval_words'] if w in word_set)
                neg_co_occurrences = sum(1 for w in definitions['negative_eval_words'] if w in word_set)
                
                aspect_score += (pos_co_occurrences - neg_co_occurrences)
                aspect_total_count += 1
        
        if aspect_total_count > 0:
            normalized_score = aspect_score / aspect_total_count
        else:
            normalized_score = 0
        
        aspect_scores[aspect_name] = normalized_score
        
    return aspect_scores

def plot_aspect_comparison(df_aspect_scores, file_name):
    """評価観点別スコアをレーダーチャートで可視化する"""
    
    categories = list(df_aspect_scores.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, color='grey', size=12)
    
    # 🌟 修正点: 目盛りの最小値を0に固定し、視覚的な比較を容易にする 🌟
    max_val = max(df_aspect_scores.values.flatten()) * 1.2
    min_val = 0 # 最小値を0に固定
    
    ax.set_rlabel_position(0)
    r_ticks = np.linspace(min_val, max_val, 5)
    plt.yticks(r_ticks, [f'{r:.1f}' for r in r_ticks], color="grey", size=10)
    plt.ylim(min_val, max_val)
    
    colors = ['#FF6347', '#4682B4', '#3CB371']
    titles = df_aspect_scores.index
    
    for i, title in enumerate(titles):
        # スコアが負の場合でも、グラフ上は0から描画されるように調整が必要なため、
        # 0未満の値を0にクリップしてプロットする (元のスコアは保持)
        values = df_aspect_scores.loc[title].values.flatten().tolist()
        plot_values = [max(0, v) for v in values] # 0未満は0でプロット
        plot_values += plot_values[:1]
        
        ax.plot(angles, plot_values, linewidth=2, linestyle='solid', label=title, color=colors[i % len(colors)])
        ax.fill(angles, plot_values, color=colors[i % len(colors)], alpha=0.25)
        
    plt.title('ゲームタイトル別 シナリオ評価観点スコア比較', size=16, y=1.1)
    ax.legend(loc='lower right', bbox_to_anchor=(1.25, 0.1))
    plt.savefig(file_name) 
    plt.close()
    
 
def main():
    print("共起分析（評価観点別スコアリング）を開始します...")

    if not os.path.exists('results'):
        os.makedirs('results')

    aspect_scores_list = []

    for config in file_config:
        title = config['title']
        path = config['path']
        review_col = config['review_col']
        
        print(f"\n==================== 📊 {title} のデータ処理を開始 ====================")
        
        df = force_read_csv(path)
        if df is None or review_col not in df.columns:
            print(f"エラー: {title}のファイル読み込みまたは列名'{review_col}'の確認に失敗しました。スキップします。")
            continue

        df_game = df.copy()
        df_game = df_game.rename(columns={review_col: 'Original_Review'})
        df_game['Original_Review'] = df_game['Original_Review'].astype(str).str.strip().replace('nan', '')
        df_game = df_game[df_game['Original_Review'].str.len() > 1].reset_index(drop=True)
        game_reviews = df_game['Original_Review'].tolist()
        
        processed_words_list = [preprocess_text(review, mecab) for review in game_reviews]
        
        # --- 観点別スコアリングの実行 ---
        scores = calculate_co_occurrence_score(processed_words_list)
        scores['Game_Title'] = title
        aspect_scores_list.append(scores)
        
        print(f"✅ {title} の観点別スコアを計算しました。")

  
    df_aspect_scores = pd.DataFrame(aspect_scores_list)
    df_aspect_scores.set_index('Game_Title', inplace=True)
    df_aspect_scores = df_aspect_scores[list(evaluation_aspects.keys())] 

    print("\n✅ 最終的な評価観点別スコア (正規化済み):")
    print(df_aspect_scores)

    # --- グラフの可視化と保存 ---

    plot_aspect_comparison(df_aspect_scores, 'results/aspect_comparison_radar_chart_optimized.png')
    print("✅ 評価観点別スコアをレーダーチャートとして保存しました。")

    output_path = 'results/aspect_scores_summary_optimized.csv'
    df_aspect_scores.to_csv(output_path, encoding='utf-8')
    print(f"✅ 観点スコアのサマリーを '{output_path}' に保存しました。")

    print("\n--- 共起分析スクリプトの全処理を完了しました ---")

if __name__ == "__main__":
    main()