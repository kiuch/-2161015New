import pandas as pd
import MeCab
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. Windows用フォント設定
# ==========================================
plt.rcParams['font.family'] = 'MS Gothic'

# ==========================================
# 1. 辞書定義（修正版を適用）
# ==========================================

# 研究定義に基づく4つの評価項目
ASPECTS = {
    "構成語": ["展開", "結末", "クライマックス", "ストーリー", "流れ", "構成", "シナリオ", "伏線"],
    "人物語": ["キャラクター", "主人公", "仲間", "登場人物", "ライバル"],
    "テーマ語": ["絆", "友情", "テーマ", "メッセージ", "人間関係", "成長"],
    "体験語": ["冒険", "探索", "旅", "世界観", "舞台", "体験"]
}

# 修正済みポジティブ辞書（カタカナ表記 + 追加語）
POSITIVE_WORDS_SET = {
   "スバラシイ", "カンドウ", "サイコウ", "メイサク", "オモシロイ", "ヨイ", "スキ", "コエル","テイネイ","コセイ","イッパイ",
    "セットクリョク", "ボツニュウ", "タカイ", "ナク", "カミ", "タノシイ", "カイシュウ","オドル",
    "キタイ", "アツイ", "カワイイ", "ツナガル", "シュウイツ", "シンセン", "リアル","ムチュウ",
    "キワダツ", "カンセイド", "マッチ", "ネッチュウ", "ヒキコマレル", "ケッサク", "ツヨイ","ブカイ",
    "ミゴト", "ワクワク", "ボリューム", "アイチャク", "イトシイ", "ナットク", "キョウカン","フカイ",
    "シッカリ", "マンゾク", "セイチョウ", "キフク", "ミリョク", "タノシム","チカイ","コウフン","hヒク",
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

class CooccurrenceAnalyzer:
    def __init__(self):
        try:
            self.tagger = MeCab.Tagger()
        except Exception as e:
            print(f"Error: MeCabの初期化に失敗しました: {e}")
            import sys
            sys.exit(1)

    def _get_tokens(self, text):
        """
        テキストから単語情報（表層形・読み・原形）のセットを抽出する
        """
        if not isinstance(text, str):
            return set()
        
        tokens = set()
        node = self.tagger.parseToNode(text)
        
        while node:
            features = node.feature.split(',')
            pos = features[0]
            
            # 名詞、形容詞、動詞、形状詞、副詞、形容動詞、助動詞 を対象
            target_pos = ["名詞", "形容詞", "動詞", "形状詞", "副詞", "形容動詞", "助動詞"]
            
            if pos in target_pos:
                surface = node.surface
                
                # ストップワード判定
                if surface in STOP_WORDS:
                    node = node.next
                    continue

                # 表層形（漢字など）を追加
                tokens.add(surface)
                
                # 原形（基本形）を追加
                if len(features) > 6 and features[6] != "*":
                    tokens.add(features[6])
                
                # 読み（カタカナ）を追加 -> これが辞書とのマッチングに重要
                if len(features) > 7 and features[7] != "*":
                    tokens.add(features[7])

            node = node.next
        return tokens

    def calculate_sentiment_counts(self, tokens):
        """抽出されたトークンセットからポジ・ネガ数をカウント"""
        pos_count = 0
        neg_count = 0
        
        # 辞書との照合（重複カウントを防ぐため、トークンセットに対して判定）
        # トークンの中にポジティブ語が含まれていればカウント
        if not tokens.isdisjoint(POSITIVE_WORDS_SET):
            # 1つのレビュー内で複数のポジティブ語があっても、今回は「単語の出現数」としてカウントするため
            # 積集合の数を見る
            pos_count = len(tokens.intersection(POSITIVE_WORDS_SET))
            
        if not tokens.isdisjoint(NEGATIVE_WORDS_SET):
            neg_count = len(tokens.intersection(NEGATIVE_WORDS_SET))
            
        return pos_count, neg_count

    def analyze(self, df, text_col):
        cooccurrence_scores = {}
        print("\n--- 共起分析を実行中 ---")
        
        # 前処理：各レビューをトークン化しておく
        df['tokens'] = df[text_col].apply(self._get_tokens)

        # 4つの評価語群（構成語・人物語・テーマ語・体験語）ごとにループ
        for aspect_name, aspect_words in ASPECTS.items():
            
            # その評価語が含まれているレビューを抽出
            # トークンセットの中に、評価語のいずれかが含まれているか
            target_mask = df['tokens'].apply(lambda t: not t.isdisjoint(set(aspect_words)))
            target_reviews = df[target_mask]
            
            total_reviews_count = len(target_reviews)
            
            if total_reviews_count == 0:
                cooccurrence_scores[aspect_name] = 0.0
                continue
                
            total_pos = 0
            total_neg = 0
            
            # 対象レビューのポジネガ数を合計
            for tokens in target_reviews['tokens']:
                p, n = self.calculate_sentiment_counts(tokens)
                total_pos += p
                total_neg += n
            
            # 正規化スコア計算: (ポジティブ総数 - ネガティブ総数) / 評価語を含むレビュー総数
            score = (total_pos - total_neg) / total_reviews_count
            cooccurrence_scores[aspect_name] = score
            
            print(f"項目[{aspect_name}]: 対象レビュー数={total_reviews_count}, Pos={total_pos}, Neg={total_neg}, Score={score:.4f}")

        return cooccurrence_scores

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

    analyzer = CooccurrenceAnalyzer()
    all_cooccurrence_scores = {}

    # ファイル設定
    file_config = [
    {'title': 'SV', 'path': r'C:\Users\masat\OneDrive\デスクトップ\deep learning\パワポ\-2161015New\SVシナリオレビュー''.csv', 'review_col': 'シナリオ小文章'},
    {'title': '剣盾', 'path': r'C:\Users\masat\OneDrive\デスクトップ\deep learning\パワポ\-2161015New\剣盾シナリオ文.csv', 'review_col': 'シナリオ一文'},
    {'title': 'USUM', 'path': r'C:\Users\masat\OneDrive\デスクトップ\deep learning\パワポ\-2161015New\sm_usumシナリオ文.csv', 'review_col': 'シナリオ文'},
    {'title': 'XY', 'path': r'C:\Users\masat\OneDrive\デスクトップ\deep learning\パワポ\-2161015New\XYシナリオ文.csv', 'review_col': 'シナリオ'}
]

    for config in file_config:
        title = config['title']
        path = config['path']
        col = config['review_col']
        
        print(f"\n========== {title} の分析を開始 ==========")
        
        df = force_read_csv(path)
        if df is None:
            print(f"エラー: {path} が読み込めませんでした。")
            continue
            
        # データクリーニング
        df = df.dropna(subset=[col])
        df[col] = df[col].astype(str).replace('nan', '')
        df = df[df[col].str.len() > 1]

        # 分析実行
        scores = analyzer.analyze(df, col)
        all_cooccurrence_scores[title] = scores

    # グラフ作成
    if all_cooccurrence_scores:
        print("\n========== グラフ作成 ==========")
        df_scores = pd.DataFrame(all_cooccurrence_scores).T
        
        # 項目の順序を定義通りに並べ替え
        column_order = ["構成語", "人物語", "テーマ語", "体験語"]
        df_scores = df_scores.reindex(columns=column_order)
        
        print(df_scores)
        
        # CSV保存
        df_scores.to_csv('results/cooccurrence_scores_final.csv', encoding='utf-8-sig')

        # 棒グラフ描画
        ax = df_scores.plot(kind='bar', figsize=(12, 6), width=0.8)
        plt.title("作品別 評価語群スコア比較 (修正版)")
        plt.ylabel("正規化スコア")
        plt.axhline(0, color='black', linewidth=0.8)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('results/cooccurrence_chart_final.png')
        plt.show()
        print("  -> グラフ保存完了: results/cooccurrence_chart_final.png")

if __name__ == "__main__":
    main()