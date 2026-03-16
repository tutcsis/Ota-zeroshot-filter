from tap import Tap
from tqdm import tqdm
from datasets import load_dataset

class Args(Tap):
  dataset_path: str = "data/toxicity_dataset_ver2.jsonl"
  toxic_folder: str = "data/LLM-jp_toxicity/toxic"
  expression_folder: str = "data/LLM-jp_toxicity/expression"

def main(args):
  dataset = load_dataset("json", data_files=args.dataset_path, split="train")
  args.tasknames = [task for task in dataset.column_names if task not in ["id", "text", "label"]]

  for category in args.tasknames:
    filtered_dataset = dataset.filter(lambda x: x[category] == "yes")
    print(f"Category: {category}, Count: {filtered_dataset.num_rows}")
    toxic_dataset = filtered_dataset.filter(lambda x: x["label"] == "toxic")
    expression_dataset = filtered_dataset.filter(lambda x: x["label"] == "has_toxic_expression")
    print(f"  Toxic Count: {toxic_dataset.num_rows}, Expression Count: {expression_dataset.num_rows}")
    toxic_dataset.to_json(f"{args.toxic_folder}/{category}.jsonl", orient="records", force_ascii=False)
    expression_dataset.to_json(f"{args.expression_folder}/{category}.jsonl", orient="records", force_ascii=False)
  

if __name__ == '__main__':
  args = Args().parse_args()
  main(args)
