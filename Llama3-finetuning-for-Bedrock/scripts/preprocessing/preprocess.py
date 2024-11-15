import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import logging

system_message = """You are Llama, an AI assistant created to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""

def convert_format(data):
    messages = [{"content": system_message, "role": "system"}]
    for item in data['dialog']:
        messages.append({"content": item['content'], "role": item['role']})
    return {"messages": messages}

def write_jsonl(data, file_path):
    print(len(data))
    try:
        print(f"Writing {file_path} file")
        with open(file_path, 'w') as f:
            item_count = 0
            for obj in data:
                json_str = json.dumps(obj)
                f.write(json_str + "\n")
                item_count += 1
        print(f"Successfully wrote {item_count} items to {file_path}")
        logging.info(f"Successfully wrote {item_count} items to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"An error occurred while writing to {file_path}: {e}")
        raise


def preprocess_data(input_data_path, output_data_path):
    # Load the data
    data = []
    with open(input_data_path) as f:
        for line in f:
            data.append(json.loads(line))
    
    # Split the data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print("Converting training data")
    dataset_train = map(convert_format,train_data)
    dataset_train=list(dataset_train)
    print("Converting test data")
    dataset_test = map(convert_format,test_data)
    dataset_test=list(dataset_test)
    print(f"converted training with len {len(dataset_train)}")
    print(f"converted training with len {len(dataset_test)}")
    # Save the processed datasets
    print("Writing JSONL file...")
    print(output_data_path+"/train")
    print(output_data_path+"/test")
    #create output_data_path if it doesn't exist
    os.makedirs(output_data_path, exist_ok=True)
    os.makedirs(output_data_path+"/train", exist_ok=True)
    os.makedirs(output_data_path+"/test", exist_ok=True)
    write_jsonl(dataset_train, output_data_path+"/train/train.json")
    write_jsonl(dataset_test, output_data_path+"/test/test.json")
    
    print("JSONL file created successfully!")
    logging.info("JSONL file created successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    input_data_path = os.path.join("/opt/ml/processing/input", "train.jsonl")
    output_data_path = "/opt/ml/processing/output"
    preprocess_data(input_data_path, output_data_path)
