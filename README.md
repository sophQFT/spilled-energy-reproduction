# Spilled Energy 再現実装 (Reproduction)

ICLR 2026 論文 **"Spilled Energy in Large Language Models"** の再現実装プロジェクトです。
LLM の最終 softmax を Energy-Based Model として解釈し、追加学習なしでハルシネーション（誤答）を検出する手法を、自分の環境で動かして検証します。

## クレジット（このリポジトリの位置づけ）

- **原論文**: Adrian R. Minut, Hazem Dewidar, Iacopo Masi.
  *Spilled Energy in Large Language Models.* ICLR 2026 / [arXiv:2602.18671](https://arxiv.org/abs/2602.18671)
- **原著者による公式実装**: https://github.com/OmnAI-Lab/spilled-energy

手法・コードの考案者は上記の原著者です。
本リポジトリは、**再現作業の記録・手順・自分の検証結果**をまとめたものであり、手法そのものの提案ではありません。

## このリポジトリで自分がやっていること

- 論文手法の理解と、公式実装のコードリーディング
- 研究室環境（単一 GPU）での再現手順の確立と文書化
- 再現結果と論文報告値の比較・考察

## 実行環境

| 項目 | 内容 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 (24GB) |
| CUDA | 12.8 |
| OS | Linux (研究室サーバー) |
| Python 環境 | uv + venv |

## 進捗ステータス

- [x] 環境構築（GPU / 仮想環境 / Hugging Face 認証）
- [ ] 公式実装のコードリーディング
- [ ] 単一サンプルでの Spilled Energy 計算の確認
- [ ] 合成算術タスク（実験1）の再現
- [ ] 実世界ベンチマーク（実験2）の再現

## ドキュメント

- [00_setup.md](notes/00_setup.md) — 環境構築の手順

## 手法の概要（自分の理解）

自己回帰 LLM では、chain rule により隣接する生成ステップ間で本来一致するはずの2つの量が存在する。

- 時刻 i: トークン x_i を生成したときの **単一 logit**
- 時刻 i+1: x_i を含む文脈から出る **語彙全体の log-sum-exp**

この2つの差が **spilled energy** であり、理論上ゼロになるはずだが実際の LLM では一致しない。
論文はこのズレが誤答と相関することを示している。訓練済み probe classifier を必要としない点が特徴。
