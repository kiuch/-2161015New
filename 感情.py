import pandas as pd
import MeCab
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'MS Gothic'
POSITIVE_WORDS_SET = {
   "スバラシイ", "カンドウ", "サイコウ", "メイサク", "オモシロイ", "ヨイ", "スキ", "コエル","テイネイ","コセイ","イッパイ",
    "セットクリョク", "ボツニュウ", "タカイ", "ナク", "カミ", "タノシイ", "カイシュウ","オドル",
    "キタイ", "アツイ", "カワイイ", "ツナガル", "シュウイツ", "シンセン", "リアル","ムチュウ",
    "キワダツ", "カンセイド", "マッチ", "ネッチュウ", "ヒキコマレル", "ケッサク", "ツヨイ","ブカイ",
    "ミゴト", "ワクワク", "ボリューム", "アイチャク", "イトシイ", "ナットク", "キョウカン","フカイ",
    "シッカリ", "マンゾク", "セイチョウ", "キフク", "ミリョク", "タノシム","チカイ","コウフン","ヒク","ヨイン","トリハダ","メチャ","ポイント",
    "サイコウホウ", "シュウバン", "ナケル", "キタイイジョウ","ハッピー","スゴイ", "ウレシイ", "アガタ", "マケ", "タベ", "ヨカッタ", "マンゾク", "サスガ"
}

# ネガティブ辞書
NEGATIVE_WORDS_SET = {
      "ヨワイ", "ヘイボン", "ザンネン", "チンプ", "サイアク", "ストレス", "ナイ","アンマリ","ヒクイ","マイル",
    "ヒョウカデキナイ", "ビミョウ", "アッサリ", "コドモムケ", "ツマラナイ", "テキトウ","ソマツ",
    "フマン", "ワルイ", "オクレ", "クソ", "モンダイ", "ソガイ", "メンドウ","ミジカイ","ナシ","イラナイ", 
    "コンナン", "ツタナサ", "モノタリナイ", "キタイハズレ", "タンチョウ", "デキナイ","ブソク",
    "フカイカン", "イミフメイ", "ウスイ", "タイクツ", "チセツ", "シリツボミ", "シリメツレツ",
    "カッテ", "フカンゼン", "アサイ", "セッキョウクサイ", "サイテイ","アキル","ウスッペライ","モノタリナイ","デキナイ","ノコラナイ",
}

# ストップワード
STOP_WORDS = {
    "ゲーム", "ポケモン", "シリーズ", "プレイ",
}

# ==========================================
# 2. 分析クラス定義
# ==========================================

class SentimentAnalyzer:
    def __init__(self):
        try:
            self.tagger = MeCab.Tagger()
        except Exception as e:
            print(f"Error: MeCabの初期化に失敗しました: {e}")
            import sys
            sys.exit(1)

    def classify_review(self, text):
        """
        1つのレビュー文を解析し、(ポジティブ数, ネガティブ数, 判定ラベル) を返す
        """
        if not isinstance(text, str):
            return 0, 0, "Neutral"
        
        pos_count = 0
        neg_count = 0
        
        node = self.tagger.parseToNode(text)
        
        while node:
            features = node.feature.split(',')
            pos = features[0]
            
            # 全方位マッチング用の品詞フィルタ
            target_pos = ["名詞", "形容詞", "動詞", "形状詞", "副詞", "形容動詞", "助動詞"]
            
            if pos in target_pos:
                surface = node.surface
                candidates = {surface}
                
                if len(features) > 6 and features[6] != "*":
                    candidates.add(features[6]) # 基本形
                if len(features) > 7 and features[7] != "*":
                    candidates.add(features[7]) # 読み

                if not candidates.isdisjoint(STOP_WORDS):
                    node = node.next
                    continue

                is_match = False
                
                if not candidates.isdisjoint(NEGATIVE_WORDS_SET):
                    neg_count += 1
                    is_match = True
                
                if not is_match and not candidates.isdisjoint(POSITIVE_WORDS_SET):
                    pos_count += 1

            node = node.next
        if pos_count > neg_count:
            sentiment = "Positive"  # 肯定的
        elif neg_count > pos_count:
            sentiment = "Negative"  # 否定的
        else:
            sentiment = "Neutral"   # 中立的 (同数または0)
            
        return pos_count, neg_count, sentiment

    def analyze_dataset(self, df, text_col):
        # データフレームの各行に対して分析を適用
        results = df[text_col].apply(lambda x: self.classify_review(x))
        
        # 結果を新しい列として追加
        df['Pos_Count'] = [res[0] for res in results]
        df['Neg_Count'] = [res[1] for res in results]
        df['Sentiment'] = [res[2] for res in results]
        
        return df

def force_read_csv(file_path):
    for encoding in ['utf-8', 'shift_jis', 'cp932']:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except:
            continue
    return None

# ==========================================
# 3. 実行メイン処理
# ==========================================

def main():
    if not os.path.exists('results'):
        os.makedirs('results')

    analyzer = SentimentAnalyzer()

    # ファイル設定
    file_config = [
    {'title': 'SV', 'path': r'C:\Users\masat\OneDrive\デスクトップ\deep learning\パワポ\-2161015New\SVシナリオレビュー''.csv', 'review_col': 'シナリオ小文章'},
    {'title': '剣盾', 'path': r'C:\Users\masat\OneDrive\デスクトップ\deep learning\パワポ\-2161015New\剣盾シナリオ文.csv', 'review_col': 'シナリオ一文'},
    {'title': 'USUM', 'path': r'C:\Users\masat\OneDrive\デスクトップ\deep learning\パワポ\-2161015New\sm_usumシナリオ文.csv', 'review_col': 'シナリオ文'},
    {'title': 'XY', 'path': r'C:\Users\masat\OneDrive\デスクトップ\deep learning\パワポ\-2161015New\XYシナリオ文.csv', 'review_col': 'シナリオ'}
]


    # グラフ描画用の設定
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # グラフの色設定
    colors = {'Positive': '#66b3ff', 'Negative': '#ff9999', 'Neutral': '#99ff99'}
    label_order = ['Positive', 'Negative', 'Neutral']

    for i, config in enumerate(file_config):
        title = config['title']
        path = config['path']
        col = config['review_col']
        
        print(f"\n========== {title} の感情分析を開始 ==========")
        
        df = force_read_csv(path)
        if df is None:
            print(f"エラー: {path} が読み込めませんでした。")
            continue
            
        # データクリーニング
        df = df.dropna(subset=[col])
        df[col] = df[col].astype(str).replace('nan', '')
        df = df[df[col].str.len() > 1]

        # 分析実行
        df_result = analyzer.analyze_dataset(df, col)
        
        # 集計
        sentiment_counts = df_result['Sentiment'].value_counts()
        
        # 存在しないラベルを0で埋める（グラフ描画のため）
        for label in label_order:
            if label not in sentiment_counts:
                sentiment_counts[label] = 0
        
        # 並び替え
        sentiment_counts = sentiment_counts[label_order]
        
        print(f"集計結果:\n{sentiment_counts}")
        
        # CSV保存
        output_csv = f'results/{title}_sentiment_details.csv'
        df_result.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"詳細データを保存しました: {output_csv}")

        # 円グラフ描画
        ax = axes[i]
        
        # データがある場合のみ描画
        if sentiment_counts.sum() > 0:
            wedges, texts, autotexts = ax.pie(
                sentiment_counts, 
                labels=sentiment_counts.index, 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=[colors[l] for l in sentiment_counts.index],
                counterclock=False,
                wedgeprops={'edgecolor': 'white'}
            )
            ax.set_title(f"{title} 感情割合")
        else:
            ax.text(0.5, 0.5, "データなし", ha='center', va='center')
            ax.set_title(f"{title} (データなし)")

    plt.tight_layout()
    plt.savefig('results/sentiment_pie_charts.png')
    plt.show()
    print("\n全処理完了: 感情分析結果の円グラフを 'results/sentiment_pie_charts.png' に保存しました。")

if __name__ == "__main__":
    main()