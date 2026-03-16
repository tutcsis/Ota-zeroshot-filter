import pandas as pd
from tap import Tap

class Args(Tap):
	csv_path: str
	out_path: str = None

def main(args):
	# csv_path = "/work/s245302/multiclassification/analysis_results/toxicity_ver2_label_ratio_all.csv"
	# csv_path = "outputs/ruri-v3-310m/2025-06-29/16-10-23.182120/log.csv"

	# print("Start converting CSV to Markdown table...")
	# print(f"CSV Path: {args.csv_path}")
	df = pd.read_csv(args.csv_path)
	columns = df.columns.tolist()

	# tranpose the dataframe
	markdown_table = f"| category | {' | '.join(columns[1:])} |\n"
	markdown_table += "| --- | " + " | ".join(["---"] * len(columns[1:])) + " |\n"

	for _, row in df.iterrows():
		category = row.iloc[0] if pd.isna(row.iloc[0]) else row.iloc[0]
		values = []
		for col in columns[1:]:
			if pd.isna(row[col]):
				values.append("")
			elif isinstance(row[col], (int, float)):
				values.append(f"{row[col]:.4f}" if isinstance(row[col], float) else str(row[col]))
			else:
				values.append(str(row[col]))
		
		markdown_table += f"| {category} | {' | '.join(values)} |\n"
  
	if args.out_path:
		with open(args.out_path, 'w', encoding='utf-8') as f:
			f.write(markdown_table)
	else:
		# print("Markdownテーブル:")
		print(markdown_table)

if __name__ == '__main__':
	args = Args().parse_args()
	csv_path = args.csv_path
	main(args)