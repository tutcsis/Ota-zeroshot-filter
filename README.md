# Ota-zeroshot-filter

- outlines を使う: https://github.com/dottxt-ai/outlines#quickstart
- schema を作成して、model に投げることで出力形式を指定する形式
- 様々なタスクの指定方法を wiki に載せてくれている
  - https://dottxt-ai.github.io/outlines/latest/features/core/output_types/#multiple-choices
  - Multiple Choices は選択肢を複数渡すことしかできず、マルチタスク問題には不適かもしれない
  - JSON Schemas は階層構造を渡すことができるのでマルチタスク問題を解決できるが、出力が安定するかがわからない?

## 実験
まず、ドキュメントを見ながら、1例を出力できるかを確かめる。

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

### ポイント
- `outlines` には `models` があり、これの中にモデルごとにライブラリがある。今回は `Transformers` を使用した
- プロンプトの最大文字数はデフォルトで 4096トークン: https://huggingface.co/llm-jp/llm-jp-3-1.8b
- 今回は、元のデータセットのテキストを1000文字使用するように変更


