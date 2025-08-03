import json

# example 1: get value from JSON string
text = '{ "obscene": "no", "discriminatory": "no", "violent": "no", "illegal": "no", "personal": "no", "corporate": "no" }'
t_text = {'id': '00001', 'label': 'nontoxic', 'obscene': 'no', 'discriminatory': 'no', 'violent': 'no', 'illegal': 'no', 'personal': 'no', 'corporate': 'no', 'others': 'no'}

data_dict = json.loads(text)
for key in data_dict:
  print(f"{key}: {data_dict[key]}")
  print(f"t_text['{key}']: {t_text[key]}")



