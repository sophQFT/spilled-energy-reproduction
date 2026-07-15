# 01. CUDA 環境のミスマッチと恒久対処

原著リポジトリの `uv.lock` をそのまま使うと，手元の GPU で PyTorch が動かない問題が発生した。
その診断と対処の記録。

---

## 症状

`uv run` でスクリプトを実行するとエラーになる。

```text
ImportError: .../torch/lib/libtorch_cuda.so: undefined symbol: ncclCommResume
```

---

## 原因

原著の `uv.lock` は torch を **2.12.1（CUDA 13 ビルド）** に固定していた。
一方，研究室サーバーの環境は次の通り。

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| Driver | 570.86.10 |
| 対応 CUDA | 12.8 |

CUDA 13 のライブラリはドライバ 580 以降を要求するため，この環境では動かない。
`torch.cuda.is_available()` は `False` になり，以下の警告が出ていた。

```text
The NVIDIA driver on your system is too old (found version 12080).
```

つまり **原著者の環境は CUDA 13 前提であり，手元の CUDA 12.8 環境と食い違っている**。

---

## 罠: `uv run` は手動修正を巻き戻す

`uv pip install` で手動で cu128 版に直しても，次に `uv run` を実行すると
lock ファイルとの同期が走り，cu130 版に戻されてしまう。

```text
Uninstalled 4 packages in 444ms
Installed 4 packages in 332ms
```

→ 手動パッチでは解決しない。**lock ファイル側を直す必要がある**。

---

## 対処: pyproject.toml で cu128 インデックスに固定

`pyproject.toml` の末尾に以下を追記した。

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
```

- `explicit = true` … 明示指定したパッケージのみこのインデックスから取得する
- `[tool.uv.sources]` … torch をこのインデックスから解決する

lock ファイルを作り直す。

```bash
uv lock
```

結果：

```text
Updated torch v2.12.1 -> v2.11.0+cu128
Removed nvidia-nccl-cu13 v2.29.7
Added nvidia-nccl-cu12 v2.28.9
```

---

## 副次的な問題: venv のメタデータ破損

`uv sync` 後も別のエラーが出た。

```text
ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory
```

調査すると，**メタデータだけ残り実体が無い**状態だった。

```bash
uv pip list | grep cudnn
# → nvidia-cudnn-cu12  9.19.0.56  （入っていることになっている）

find .venv -name "libcudnn.so*"
# → 0 件（実体が無い）

ls .venv/lib/python3.11/site-packages/nvidia/
# → cudnn ディレクトリが存在しない

ls -d .venv/lib/python3.11/site-packages/nvidia_cudnn*
# → nvidia_cudnn_cu12-9.19.0.56.dist-info （メタデータのみ）
```

cu13 → cu12 の入れ替え過程でアンインストールが中途半端に終わり，
dist-info だけが残ったと考えられる。この状態では `uv sync` が
「既にインストール済み」と判断してスキップするため，永久に修復されない。

### 解決: venv を作り直す

```bash
cd ~/paper-replication/spilled-energy
```

```bash
pwd
```

※ `rm -rf` の前に必ず現在地を確認する

```bash
rm -rf .venv
```

```bash
uv sync
```

```bash
source .venv/bin/activate
```

`.venv` は `uv.lock` から再生成できるので削除して問題ない（`.gitignore` にも登録済み）。

---

## 確認

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

成功時の出力：

```text
2.11.0+cu128
12.8
True
NVIDIA GeForce RTX 4090
```

---

## 教訓

- 原著の lock ファイルは，原著者の GPU 環境を前提にしている。ドライバが違えば動かない
- `uv run` / `uv sync` は lock と同期するため，**手動パッチは無意味**。lock 側を直す
- パッケージ管理が壊れたら，継ぎ足しで直すより venv を作り直すほうが確実
- `uv pip list` は dist-info を見ているだけで，実体があるとは限らない

---

## 補足: uv.lock は git 管理外

原著リポジトリの `.gitignore` に `*.lock` が含まれているため，
`uv.lock` は git に追跡されていない。

```bash
git ls-files uv.lock
# → 何も出力されない（追跡されていない）
```

```bash
git check-ignore -v uv.lock
# → .gitignore:12:*.lock    uv.lock
```

そのため **lock ファイルの変更はコミットできない**。
環境固定は `pyproject.toml` 側（`[tool.uv.sources]`）で行う必要がある。

clone した人が `uv lock` を実行すれば，pyproject.toml の指定に従って
cu128 版に解決されるので，設定の意図は共有される。
