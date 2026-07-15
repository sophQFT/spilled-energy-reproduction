# 00. 環境構築 / 作業再開の手順

このプロジェクトでは **2つのリポジトリ** を使い分ける。

| リポジトリ | 役割 | パス |
|---|---|---|
| `spilled-energy` | 原著コード（fork）。Python を実行する側 | `~/paper-replication/spilled-energy` |
| `spilled-energy-reproduction` | 本リポジトリ。README・ノート・結果 | `~/paper-replication/spilled-energy-reproduction` |

---

## 1. サーバーへ接続（接続先・ユーザー名は各自環境に置き換え）

```bash
ssh <username>@<server-address>
```

例：

```bash
ssh username@192.168.1.xx
```

---

## 2. 原著コードの取得（初回のみ）

置き場所となる親ディレクトリを作る。

```bash
mkdir -p ~/paper-replication
```

親ディレクトリへ移動する。

```bash
cd ~/paper-replication
```

原著リポジトリを clone する（自分は fork 版を使用）。

```bash
git clone https://github.com/sophQFT/spilled-energy.git
```

参考：本家は以下。

```bash
git clone https://github.com/OmnAI-Lab/spilled-energy.git
```

---

## 3. 本リポジトリ（ノート側）の取得（初回のみ）

```bash
cd ~/paper-replication
```

```bash
git clone https://github.com/sophQFT/spilled-energy-reproduction.git
```

---

## 4. 作業ブランチを確認

原著コード側は `replication` ブランチで作業する。

```bash
cd ~/paper-replication/spilled-energy
```

現在のブランチを確認する（`*` が付いているものが現在地）。

```bash
git branch
```

`replication` に切り替える場合：

```bash
git switch replication
```

---

## 5. ログイン後の定型作業

サーバーにログインしたら，まず以下を実行する。

```bash
cd ~/paper-replication/spilled-energy
```

```bash
source .venv/bin/activate
```

仮想環境が有効になると，プロンプトの先頭に `(spilled-energy)` が表示される。

---

## 6. 作業前に Git の状態を確認

変更途中のファイルがないか確認する。

```bash
git status
```

GitHub 側の最新状態を取り込む場合：

```bash
git pull
```

---

## 7. GPU を確認する

GPU を使う実験の前に確認する。

```bash
nvidia-smi
```

RTX 4090 が認識され，`Memory-Usage` が空いていることを確認する。

---

## 8. Hugging Face ログイン（初回のみ）

```bash
huggingface-cli login
```

- トークンは https://huggingface.co/settings/tokens で **Type: Read** として作成する
- 貼り付けても画面には何も表示されないが正常（そのまま Enter）
- `Add token as git credential?` は `n` で良い
- Llama-3 8B Instruct は gated モデルのため，HF 上で事前にアクセス申請が必要

---

## 9. メモを編集する

ノートは reproduction 側にあるので，移動してから編集する。

```bash
cd ~/paper-replication/spilled-energy-reproduction
```

```bash
nano notes/00_setup.md
```

nano の保存・終了：

```text
Ctrl + O   保存
Enter      保存確定
Ctrl + X   終了
```

---

## 10. 変更を GitHub に反映する

変更内容を確認する。

```bash
git status
```

変更したファイルをステージする。

```bash
git add notes/00_setup.md
```

すべての変更をまとめてステージする場合：

```bash
git add .
```

コミットする（`-m` の後がコミットメッセージ）。

```bash
git commit -m "Add setup notes"
```

GitHub に送る。

```bash
git push
```

---

## 11. 作業再開時の最小手順

久しぶりに作業を再開するときは，基本的にこれを実行する。

```bash
ssh <username>@<server-address>
```

```bash
cd ~/paper-replication/spilled-energy
```

```bash
source .venv/bin/activate
```

```bash
git status
```

必要に応じて：

```bash
git pull
```

```bash
nvidia-smi
```

---

## 次にやること

- [ ] `src/` のコード構造を読む（generation / extraction / energy の3ファイル）
- [ ] 動作確認スクリプト `test_measure_exact_answer.py` を実行
