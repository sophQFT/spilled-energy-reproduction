# 02. energy.py の解読とテストスクリプトの検証

コアモジュール `src/spilled_energy/energy.py` を読み，論文の式との対応を確定させた。
その過程で公式のテストスクリプト `test_measure_exact_answer.py` に2つの問題を発見し，
検証実験（`experiments/01_alignment_check`, `02_convention_check`）で実証した。

- 実行日: 2026-07-16（初回検証は 2026-07-06，両日の出力は完全一致）
- モデル: meta-llama/Meta-Llama-3-8B (bfloat16, greedy)
- 質問: "Who wrote the play Romeo and Juliet?"

---

## 1. energy.py の構造

`spilled_energy(logits, ids, beta)` が本体。3つの量を返す。

```python
E_ji     = -beta * logits[j][i-1][ids[j][i]]      # E: トークンエネルギー
E_margin = -logsumexp(beta * logits[j][i])         # E_margin: 周辺エネルギー
delta    = -E_margin + E                           # ΔE: spilled energy
```

展開すると：

```text
ΔE(x_i) = logsumexp(θ(x_i:1)) − θ(x_{i−1:1})[id(x_i)]
```

これは論文 Eq. (8) と完全に一致する（スライド p.11 の式と同じ）。

### 確定事項: 符号の規約

- **符号付き**であり，絶対値は取らない（`introduction.md` の `|E − F|` は不正確な簡略記述）
- 実際，正しい計算では ΔE は負の値も取る（後述の検証で確認）

### 注意点 1: 先頭はプレースホルダ

```python
E_j = [0]  # E(x_0) = 0
for i in range(1, len(logits[j])):
```

配列先頭のトークンは「前ステップの logit」が存在しないため E=0 で埋められる。
その位置の delta は `−E_margin + 0 = LSE そのもの` になり，**spilled energy ではない**。

### 注意点 2: スライスしてから渡してはいけない

ΔE の計算には「トークン i の logit（ステップ i）」と「LSE（ステップ i+1）」の
**隣接2ステップのペア**が必要。answer span だけスライスした logits を渡すと，
スライス先頭のトークンは相方を失いプレースホルダ化する。
長さ1のスライスでは全要素が無意味になる。

→ **全系列で計算してから，delta を span で切り出す**のが正しい順序。

### energy.py が仮定する入力規約

`E[i] = logits[i−1][ids[i]]` という添字は，
**forward-pass 規約**（`logits[k]` = ids[0..k] を文脈とした出力，ids[0]=BOS）を仮定している。

---

## 2. テストスクリプトの問題点（2件）

### 問題 A: exact answer span の取りこぼし

生成文が先頭スペース付き（` William Shakespeare ...`）のため `start_idx=1` となり，
span 決定条件 `if s >= start_idx` が offset (0,8) の ' William' トークンを弾く。

```text
Extracted answer: 'William Shakespeare'
Token span: 1-2
Corresponding text from tokens:  Shakespeare   ← William が落ちる
```

### 問題 B: generate() の scores は規約が1ステップずれている

HF の `generate()` が返す `scores[t]` は「トークン t を**生成するとき**の分布」
（文脈は ids[0..t−1]）。energy.py が仮定する「トークン t を**読んだ後**の出力」とは
1ステップずれている。テストスクリプトは scores をそのまま渡している。

---

## 3. 検証実験の結果

### 3.1 alignment_check（問題 A と 93 vs 94 ズレの検証）

```text
[2] n_generated=94, last_id=128001, eos_id=128001
[3] retokenized=93, generated=94
[4] first mismatch index: None
    -> 差分は末尾のみ: [128001] = ['<|end_of_text|>']
[5] char span: 1-20
[6] 現行span: 1-2 -> ' Shakespeare'
[7] 修正span : 0-2 -> ' William Shakespeare'
[8] generated_ids側 decode: ' William Shakespeare'
```

**結論:**
- 93 vs 94 のズレは末尾 EOS のみ。EOS はテキストに現れないため再トークン化で消えるだけで，**実害なし**
- span バグは再現し，重なり条件 `e > start_idx and s < end_idx` への修正で解決する

### 3.2 convention_check（問題 B の検証）

teacher forcing で全系列の logits を再計算し，scores と突き合わせた。

```text
[1] input_len=13, n_steps=94, total_len=107
[2] max|scores[t] - logits_fp[input_len+t-1]| = 1.8633
    argmax一致: 93/94
```

→ scores[t] は forward pass の位置 input_len+t−1 と一致（bfloat16 誤差の範囲）。
**1ステップずれ仮説が確定。**

```text
[4] greedy検算 max|E_fwd[i] - (-max scores[i])| = 0.2500
```

→ greedy なら E = −max(scores) になるはずという恒等式も成立。
インデックスの解釈が正しいことの裏付け。

### 3.3 現行方式と正しい方式の数値比較

```text
[3] 生成トークン先頭8個 (i: token | E_cur | E_fwd | Δ_cur | Δ_fwd)
    0:      'ĠWilliam' |    0.000 |  -19.000 |   19.547 |    3.146
    1:  'ĠShakespeare' |  -17.125 |  -22.125 |    5.021 |   -2.670
    2:        'Ġwrote' |  -12.438 |  -18.375 |    6.983 |    3.888
    3:          'Ġthe' |  -11.750 |  -21.875 |   10.512 |    0.285
    4:         'Ġplay' |  -15.750 |  -22.125 |    6.409 |   -0.759
    5:        'ĠRomeo' |  -13.688 |  -21.125 |    7.683 |    3.642
    6:          'Ġand' |  -14.938 |  -24.750 |    9.830 |   -1.498
    7:       'ĠJuliet' |  -16.750 |  -23.250 |    6.502 |   -2.486

[6] delta_current: mean=9.7631 min=4.8825 max=19.5474
    delta_forward: mean=0.1825 min=-5.7775 max=7.6253
    max|delta_cur - delta_fwd| = 22.7772
```

**結論:**
- 現行方式（scores 直渡し）の ΔE は正値に大きく偏り，**別物の数値**になる
- 正しい forward-pass 方式では ΔE は **0 近辺に分布**（mean 0.18，負値もある）
- 正答例（William Shakespeare）で ΔE が小さいのは，論文の
  「truthful なら低い spill」（Fig. 1）と整合する

### 3.4 exact answer トークンの正しい値（forward-pass 方式）

```text
'ĠWilliam':     E=-19.0000, E_margin=-22.1462, delta= 3.1462
'ĠShakespeare': E=-22.1250, E_margin=-19.4552, delta=-2.6698
```

---

## 4. 正しい計算手順（まとめ）

1. `generate()` で回答を生成し，全系列 `sequences` を得る
2. `model(sequences)` の teacher forcing で**全系列の logits** を再計算する
3. **全系列の logits と全系列の ids** を `spilled_energy()` に渡す
4. 得られた delta 列から，exact answer span（重なり条件で決定）を**後から**切り出す

---

## 5. 教訓

- ドキュメント（introduction.md）とコードは食い違うことがある。**規約はコードと検証実験で確定させる**
- HF `generate()` の scores と forward pass の logits は**インデックス規約が1ステップ違う**
- スコアを series でスライスする処理は，隣接ステップに依存する量（ΔE）と相性が悪い。全系列で計算してから切り出す
- greedy 生成は決定的なので，同一環境なら完全に同じ出力が得られる（7/6 と 7/16 のログが一致，環境再構築後の再現性確認になった）
