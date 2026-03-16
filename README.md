# Ota-zeroshot-filter について
- outlines を使う: https://github.com/dottxt-ai/outlines#quickstart
- schema を作成して、model に投げることで出力形式を指定する形式
- 様々なタスクの指定方法を wiki に載せてくれている
  - https://dottxt-ai.github.io/outlines/latest/features/core/output_types/#multiple-choices
  - Multiple Choices は選択肢を複数渡すことしかできず、マルチタスク問題には不適かもしれない
  - JSON Schemas は階層構造を渡すことができるのでマルチタスク問題を解決できるが、出力が安定するかがわからない?

# ファイル・フォルダ構成
- .venv: python のライブラリのバージョンを管理するのに使用。ここは触らない。
- data: 取ってきたデータ
  - LLM-jp_toxicity: 有害テキストデータセット『LLM-jp Toxicity Dataset v2』を成形したデータ
    - expression: `label = has_toxic_expression` として割り当てられたデータのみを収集。有害ラベルごとに `XX.jsonl` の形式のファイルに保存。
    - toxic: `label = toxic` として割り当てられたデータのみを収集。有害ラベルごとに `XX.jsonl` の形式のファイルに保存。
- log: `qsub XX.sh` でジョブを実行したときに、生成されるジョブのログファイルを格納
- outputs: `qsub XX.sh` でモデルを学習するジョブを実行したときに生成されるモデルを自動で格納
- src: 実行する python コードを保存
  - split_toxicity_dataset.py: LLM-jp Toxicity Dataset v2 をカテゴリとtoxic/has_toxicity_expression ごとに分けて jsonl ファイルに保存
  - zeroshot_classify.py: outlines を用いて zeroshot を行い、精度を計算


- zeroshot_classify.sh: zerosho_classify.py を実行
- venv.sh: Singularity 環境に入る
- pyproject.toml: uv 環境にインストールする python ライブラリを指定

# 実験
まず、ドキュメントを見ながら、1例を出力できるかを確かめる。
ソースコード: `src/outline_sample.py`

1. JSONLデータセットからテキストとラベルを取得
2. スキーマを用意
3. モデルを用意して、テキストを投げる
4. 生成されたラベルと正解ラベルを両方出力

```
{ "toxicity": "toxic", "obscene": "no", "discriminatory": "no", "violent": "no", "illegal": "no", "personal": "no", "corporate": "no", "others": "no" }
```

- 正解は obscene のみ yes であり、今回の場合は間違った出力をしている

- ひとまず、1例のラベルを予測することはできた。
- 次は、JSONLデータセットから複数例を出力してみる

## ポイント
- `outlines` には `models` があり、これの中にモデルごとにライブラリがある。今回は `Transformers` を使用した
- プロンプトの最大文字数はデフォルトで 4096トークン: https://huggingface.co/llm-jp/llm-jp-3-1.8b
- 今回は、元のデータセットのテキストを1000文字使用するように変更

# 実験2
次に、LLM-jp Toxicity Dataset v2 を使用して、ラベルを付与させ、実際のラベルとの精度を出す
ソースコード: `src/zeroshot_classify.py`

LLMが出力した結果からJSONデータに変換する


- モデルを `llm-jp/llm-jp-3.1-1.8b-instruct4` から `llm-jp/llm-jp-3-7.2b-instruct3` に変更
- それに合わせて、メモリを 64GB から 128GB に変更
- それでもダメなら量子化する
