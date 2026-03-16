import torch
import json
import numpy as np
import utils
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datetime import datetime
from pathlib import Path
from tap import Tap
from typing import Literal
from collections import Counter
from tqdm import tqdm
from peft import prepare_model_for_kbit_training

from outlines.types import JsonSchema
from outlines.models import Transformers


class Args(Tap):
  data_path: str = "data/toxicity_dataset_ver2.jsonl"
  # max_length: int = 128
  # max_length: int = 3072
  # max_length: int = 2048
  max_length: int = 1024

  # model_name: str = "llm-jp/llm-jp-3.1-1.8b-instruct4"
  # model_name: str = "llm-jp/llm-jp-3-7.2b-instruct3"
  model_name: str = "llm-jp/llm-jp-3.1-13b-instruct4"
  output_dir: str = ""
  
  seed: int = 42

  use_bf16: bool = False
  use_fp16: bool = True

  device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
  debug: bool = True

  @property
  def torch_dtype(self):
    if self.use_bf16:
      return torch.bfloat16
    elif self.use_fp16:
      return torch.float16
    else:
      return torch.float32

  def process_args(self):
    if self.output_dir:
      self.output_dir = Path( self.output_dir )
    else:
      basename = self.model_name.split("/")[-1]
      date, time = datetime.now().strftime("%Y-%m-%d/%H-%M-%S.%f").split("/")
      output_dir = Path("outputs", basename, date, time)
      self.output_dir = output_dir
    self.output_dir.mkdir(parents=True)
    log_path = Path( self.output_dir, "parameters.txt" )
    self.log_file = log_path.open( mode='w', buffering=1 )
    print( json.dumps({
      "dataset_path": self.data_path,
      "model_name": self.model_name
    }), file=self.log_file )

  def log(self, metrics: dict) -> None:
    print("metrics: ", metrics)
    print("tasknames: ", self.tasknames)
    log_file = self.output_dir / f"log.csv"
    for category in self.tasknames:
      category_metrics = {
        "category": category,
        "accuracy": metrics[category].get(f"accuracy", -1),
        "precision": metrics[category].get(f"precision", -1),
        "recall": metrics[category].get(f"recall", -1),
        "f1": metrics[category].get(f"f1", -1),
        "TN": metrics[category].get("TN", -1),
        "FP": metrics[category].get("FP", -1),
        "FN": metrics[category].get("FN", -1),
        "TP": metrics[category].get("TP", -1),
      }
      utils.log(category_metrics, log_file)
      tqdm.write(
        f"category: {category} \t"
        f"accuracy: {category_metrics[f'accuracy']:.4f} \t"
        f"precision: {category_metrics[f'precision']:.4f} \t"
        f"recall: {category_metrics[f'recall']:.4f} \t"
        f"f1: {category_metrics[f'f1']:.4f} \t"
        f"TN: {category_metrics[f'TN']} \t"
        f"FP: {category_metrics[f'FP']} \t"
        f"FN: {category_metrics[f'FN']} \t"
        f"TP: {category_metrics[f'TP']}"
      )
      print(category, "ok!!")

def main(args):
  # 1. Process arguments
  def truncate_text(example):
    example["text"] = example["text"][:args.max_length]
    return example

  # 1.1 Load dataset
  test_dataset = load_dataset("json", data_files=args.data_path, split="train")
  print(test_dataset)
  test_dataset = test_dataset.map(truncate_text)
  test_dataset = test_dataset.remove_columns(["label", "others"])
  args.tasknames = [task for task in test_dataset.column_names if task not in ["id", "text"]]

  # 1.2 Schema and prompt
  schema = {
    "type": "object",
    "properties": {
      "obscene": {"type": "string", "enum": ["yes", "no"]},
      "discriminatory": {"type": "string", "enum": ["yes", "no"]},
      "violent": {"type": "string", "enum": ["yes", "no"]},
      "illegal": {"type": "string", "enum": ["yes", "no"]},
      "personal": {"type": "string", "enum": ["yes", "no"]},
      "corporate": {"type": "string", "enum": ["yes", "no"]},
    },
    "required": [
      "obscene", "discriminatory", "violent", "illegal", "personal", "corporate"
    ]
  }
  output_type = JsonSchema(schema)
    
  # 2. Load model and tokenizer
  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=args.torch_dtype,
  )
  model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    torch_dtype=args.torch_dtype,
    quantization_config=quantization_config
  )
  model = prepare_model_for_kbit_training(model)
  model.eval()
  ol_model = Transformers(model, tokenizer)

  # 3. Predict
  predictions = []
  labels = []
  print(test_dataset.num_rows, "rows in the test dataset")
  # for curr_idx in range(100):
  max_context_len = 0
  for curr_idx in range(test_dataset.num_rows):
    text = test_dataset[curr_idx]["text"]
    prompt = f"""
以下のテキストに対して、各項目を分類してください。

# テキスト:
「{text}」

# 各ラベルの定義:
- obscene: 露骨な性的表現やわいせつな内容を含むか？ (yes / no)
- discriminatory: さまざまな差別的表現や他者への侮辱の表現を含むか？ (yes / no)
- violent: 暴力的表現や脅迫の表現を含むか？ (yes / no)
- illegal: 違法、準法規的、非倫理的行為を助長する表現の有無を含むか？ (yes / no)
- personal: 個人情報やプライバシーの暴露を含むか？ (yes / no)
- corporate: 企業や団体の様々な機密情報の公開を含むか？ (yes / no)

# 分類結果（JSON形式で出力してください）:
"""

    result = ol_model(prompt, output_type=output_type, max_new_tokens=200)
    context_len = len(tokenizer.encode(prompt)) + len(tokenizer.encode(result))
    max_context_len = max(max_context_len, context_len)
    result_dict = json.loads(result)
    # print(f"Index: {curr_idx}")
    # print(f'prompt: {prompt}')
    # print(f'predict: {result}')
    # print('correct: ', {k: v for k, v in test_dataset[curr_idx].items() if k != "text"})
    # print("---")
  # { "obscene": "yes" , "discriminatory": "yes", "violent": "yes", "illegal": "yes", "personal": "yes", "corporate": "yes" }

    # 3.1 Make label list
    curr_predictions = []
    curr_labels = []
    for taskname in args.tasknames:
      curr_predictions.append(1 if result_dict[taskname] == "yes" else 0)
      curr_labels.append(1 if test_dataset[curr_idx][taskname] == "yes" else 0)
    predictions.append(curr_predictions)
    labels.append(curr_labels)
  print(f"max_context_len: {max_context_len}")

  # 4. Calc metrics
  def compute_metrics(predictions, labels):
    # print(f"predictions: {predictions}")
    # print(f"labels: {labels}")
    predictions = np.array(predictions).T
    labels = np.array(labels).T
    metrics = {}
    if args.debug:
      print("tasknames: ", args.tasknames)
    for taskname, _predictions, _labels in zip(args.tasknames, predictions, labels):
    # for taskname, _predictions, _labels in zip(tasknames, predictions, labels):
      label_counts = np.bincount(_labels.astype(int))

      cm = confusion_matrix(_labels, _predictions, labels=[0, 1])
      tn, fp, fn, tp = cm.ravel()
      if args.debug:
        print(f"Confusion Matrix for {taskname}:\n{cm}")
        # print(f"{taskname} answer count (yes: {label_counts[0]}), (no: {label_counts[1]})")
      if cm.shape != (2, 2):
        print(f"Warning: Confusion matrix for {taskname} is not 2x2. It may not be binary classification.")
      metrics[taskname] = {
        "accuracy": accuracy_score(_labels, _predictions),
        "precision": precision_score(_labels, _predictions, zero_division=0),
        "recall": recall_score(_labels, _predictions, zero_division=0),
        "f1": f1_score(_labels, _predictions, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
      }
    return metrics

  metrics = compute_metrics(predictions, labels)
  print(metrics)
  args.log(metrics)
  print(f"Metrics saved to, {args.output_dir} / log.csv")

if __name__ == "__main__":
  args = Args().parse_args()
  main(args)
