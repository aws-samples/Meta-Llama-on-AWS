import json
import pandas as pd
from sklearn.model_selection import train_test_split

def data_transform(row):
    content = "Instruction: {} \n Context: {}".format(row['instruction'], row['context'])
    prompt = {
        "prompt": content,
        "response": row["response"]
        
    }
    return prompt
    
data = []

with open('databricks-dolly-15k.jsonl', 'r') as f:
    for line in f:
        row = json.loads(line)
        trans_data = data_transform(row)
        data.append(trans_data)
    
df = pd.DataFrame(data)

train, test = train_test_split(df, test_size=0.2,random_state = 42)


train.to_json("dataset/train_dataset.json", orient="records", force_ascii=False)
test.to_json("dataset/test_dataset.json", orient="records", force_ascii=False) 
