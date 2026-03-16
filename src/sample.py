import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentencepiece import SentencePieceProcessor

# example 1: get value from JSON string
text = '{ "obscene": "no", "discriminatory": "no", "violent": "no", "illegal": "no", "personal": "no", "corporate": "no" }'
t_text = {'id': '00001', 'label': 'nontoxic', 'obscene': 'no', 'discriminatory': 'no', 'violent': 'no', 'illegal': 'no', 'personal': 'no', 'corporate': 'no', 'others': 'no'}

data_dict = json.loads(text)
# for key in data_dict:
#   print(f"{key}: {data_dict[key]}")
#   print(f"t_text['{key}']: {t_text[key]}")

model_name: str = "llm-jp/llm-jp-3.1-13b-instruct4"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print('AutoTokenizer: ', tokenizer)

sp = SentencePieceProcessor("models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model")
print('SentencePiece: ', sp)
