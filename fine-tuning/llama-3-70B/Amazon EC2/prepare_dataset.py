from sklearn.model_selection import train_test_split
import json
import pandas as pd

def generate_prompt(row):
    prompt = f"Instruction: {row['instruction']}\nContext: {row['context']}\nResponse: {row['response']}"
    return prompt

data = []
with open('databricks-dolly-15k.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

df['text'] = df.apply(generate_prompt, axis=1)

train, test = train_test_split(df, test_size=0.2,random_state = 42)


train.to_json("train_dataset.json", orient="records", force_ascii=False)
test.to_json("test_dataset.json", orient="records", force_ascii=False)
