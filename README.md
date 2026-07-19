# Spilled Energy in LLMs — 再現実装ログ

ICLR 2026 論文 **"Spilled Energy in Large Language Models"**（Minut et al.）の再現プロジェクト。
LLM の softmax を Energy-Based Model として読み，追加学習なしで誤答・ハルシネーションを検出する手法を，研究室の単一 GPU 環境で検証しています。

**現時点の主な成果:**
コア実装 `energy.py` を論文の式レベルで解読し，公式リポジトリのサンプルスクリプトに**2つの問題**（answer span の取りこぼし・logits のインデックス規約ズレ）があることを検証実験で実証。正しい計算手順を確立しました。詳細は [notes/02](notes/02_code_reading.md)。

---

## クレジット（このリポジトリの位置づけ）

- **原論文**: Adrian R. Minut, Hazem Dewidar, Iacopo Masi.
  *Spilled Energy in Large Language Models.* ICLR 2026 / [arXiv:2602.18671](https://arxiv.org/abs/2602.18671)
- **公式実装**: https://github.com/OmnAI-Lab/spilled-energy

手法・コードの考案者は原著者です。本リポジトリは**再現作業の記録・検証・自分の考察**をまとめたものであり，手法の提案ではありません。原著コードへの変更は fork（[sophQFT/spilled-energy](https://github.com/sophQFT/spilled-energy) の `replication` ブランチ）で管理しています。

## 関連資料

- **論文解説スライド PDF（全27枚）**: [spilled-energy-llm.pdf](https://sophqft.github.io/ml-math-learning-notes/slides/paper-reading/spilled-energy-llm/spilled-energy-llm.pdf)
  — 本論文の数理（chain rule から Eq.(8) の導出まで）を段階的に整理した発表資料
  （[ソースフォルダ](https://github.com/sophQFT/ml-math-learning-notes/tree/main/slides/paper-reading/spilled-energy-llm) / [ポートフォリオサイト](https://sophqft.github.io/ml-math-learning-notes/)）

## リポジトリ構成

```text
~/paper-replication/
├── spilled-energy/                  # 原著コードの fork（実行側）
│   └── replication ブランチ         #   └ 環境修正などの変更を記録
└── spilled-energy-reproduction/     # 本リポジトリ（記録側）
    └── notes/                       #   └ 手順・診断・コード解読ノート
```

## ドキュメント

| ノート | 内容 |
|---|---|
| [00_setup.md](notes/00_setup.md) | 環境構築と作業再開の定型手順 |
| [01_cuda_troubleshooting.md](notes/01_cuda_troubleshooting.md) | 原著 lock（CUDA 13 前提）と手元環境（CUDA 12.8）のミスマッチ診断，torch を cu128 に恒久固定した記録 |
| [02_code_reading.md](notes/02_code_reading.md) | `energy.py` と論文 Eq.(8) の対応，公式テストスクリプトの2つの問題の発見と検証実験 |

## 主な発見

1. **環境ミスマッチの恒久対処** — 原著の `uv.lock` は torch 2.12.1+cu130（CUDA 13）を固定しており，ドライバ 570.x（CUDA 12.8）環境では動かない。`pyproject.toml` の `[tool.uv.sources]` で cu128 ホイールに固定して解決（[notes/01](notes/01_cuda_troubleshooting.md)）
2. **符号規約の確定** — 実装の ΔE = LSE − logit は論文 Eq.(8) と一致（符号付き）。ドキュメント上の絶対値定義は簡略記述で，実装・論文とは異なる（[notes/02](notes/02_code_reading.md)）
3. **公式サンプルスクリプトの2つの問題** — (A) answer span 決定条件が先頭トークンを取りこぼす，(B) HF `generate()` の scores は `energy.py` が仮定する規約から1ステップずれる。teacher forcing との突き合わせで実証し，ΔE が別物の値になることを数値で確認（max 誤差 22.8）
4. **正しい計算手順の確立** — 全系列を forward pass で再計算 → 全系列で ΔE を計算 → answer span を重なり条件で後から切り出す。この方式では正答例の ΔE が 0 近辺に分布し，論文の理論と整合
5. **再現性の確認** — 環境再構築の前後で検証スクリプトの数値出力が完全一致（greedy 生成の決定性）

## 実行環境

| 項目 | 内容 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 (24GB) |
| Driver / CUDA | 570.86.10 / 12.8 |
| PyTorch | 2.11.0+cu128（cu128 インデックスに固定） |
| モデル | meta-llama/Meta-Llama-3-8B (bfloat16) |
| パッケージ管理 | uv |

## 進捗

- [x] 環境構築（GPU / venv / Hugging Face 認証）
- [x] CUDA 環境ミスマッチの診断と恒久対処
- [x] コア実装（energy.py）のコードリーディングと論文式との対応づけ
- [x] 公式テストスクリプトの問題特定と検証実験
- [ ] 修正版パイプラインの実装（span 修正 + forward-pass 方式）
- [ ] 合成算術タスク（論文 実験1 / Fig. 3）の再現
- [ ] 実世界ベンチマークの再現（※ 論文 Table 1 は別リポジトリ LLMsKnow 依存のため範囲を検討中）

## 手法の概要（自分の理解）

### 1. 出発点: chain rule と softmax

自己回帰 LLM は系列の同時確率を条件付き確率の積に分解し，各ステップを語彙上の softmax 分類器で実装している。

```text
p(x_t | x_{t-1:1}) = exp(θ(x_{t-1:1})[id(x_t)]) / Σ_k exp(θ(x_{t-1:1})[k])
```

分子は「生成されたトークンの単一 logit」，分母は「語彙全体の正規化項」である。

### 2. EBM としての読み替え

EBM の流儀（低エネルギー = 高確率）で softmax を読むと，2つのエネルギーが取り出せる。

```text
logit energy:    E^ℓ(x_t:1)   = −θ(x_{t-1:1})[id(x_t)]      … 分子由来
marginal energy: E^m(x_{t-1:1}) = −log Σ_k exp(θ(x_{t-1:1})[k]) … 分母由来
```

chain rule で系列確率を展開すると，**同じ prefix x_t:1 のエネルギーが隣接2ステップで二重に測られる**ことが分かる。時刻 t では logit として，時刻 t+1 では log-sum-exp として。理論上この2つは一致すべきだが，学習時にこの整合性は明示的に強制されていない。

### 3. Spilled Energy（論文 Eq. 8）

その不一致を spilled energy と定義する。

```text
ΔE(x_t:1) = LSE(θ(x_t:1)) − θ(x_{t-1:1})[id(x_t)]
```

- **符号付き**の量（絶対値ではない）。整合していれば ΔE ≈ 0
- 実験的に，正答は低い ΔE，誤答は高い ΔE に分布する（論文 Fig. 3）

### 4. 実用上のポイント

- **exact answer token に絞る**: 句読点や文頭語は次トークン候補が自然に広く false positive の原因になるため，意味的に答えを担う span [u, w] だけを評価する。これで平均 AuROC が約 24pt 改善（論文 Table 2）
- **pooling**: 答えが複数トークンのときは span 内で min pooling が全体的に良い傾向
- **training-free**: probe classifier のような追加学習・タスク別チューニングが不要で，cross-dataset 汎化に強い（論文 Fig. 4, Table 1）
- **限界**: white-box な logit アクセスが必要。外部事実検証ではないため，真偽の証明ではなく内部異常スコアとして使う
