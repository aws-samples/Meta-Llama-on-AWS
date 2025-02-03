import os
import subprocess
import sys
import os
import json
import torch
import logging
import argparse
import pkg_resources
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig,pipeline
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Set up logging
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)

# Anthropic/Vicuna like template without the need for special tokens
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)



def print_library_versions():
    libraries = [
        "datasets",
        "transformers",
        "accelerate",
        "evaluate",
        "bitsandbytes",
        "huggingface_hub",
        "trl",
        "peft",
        "torch",
        "scikit-learn"
    ]
    
    logger.info(f"Python version: {sys.version}")
    logger.info("Library versions:")
    for lib in libraries:
        try:
            version = pkg_resources.get_distribution(lib).version
            logger.info(f"{lib}: {version}")
        except pkg_resources.DistributionNotFound:
            logger.info(f"{lib}: Not installed")

def install_requirements(requirements_file):
    if not os.path.exists(requirements_file):
        logger.warning(f"Requirements file not found at {requirements_file}. Skipping installation.")
        return
    try:
        logger.info(f"Installing requirements from {requirements_file}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        logger.info("Successfully installed requirements")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements. Error: {e}")
        # Consider whether you want to exit here or continue
        # sys.exit(1)

def get_data_path(channel):
    processing_path = os.environ.get(f'SM_PROCESSING_OUTPUT_{channel.upper()}')
    if processing_path:
        return processing_path
    
    training_path = os.environ.get(f'SM_CHANNEL_{channel.upper()}')
    if training_path:
        return training_path
    
    default_paths = {
        'train': '/opt/ml/input/data/train',
        'test': '/opt/ml/input/data/test',
        'validation': '/opt/ml/input/data/validation'
    }
    return default_paths.get(channel, f'/opt/ml/input/data/{channel}')

def load_model_and_tokenizer(model_path):
    logger.info(f"Attempting to load model from: {model_path}")
    logger.info(f"Directory contents: {os.listdir(model_path)}")
    
    try:
        #quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        if os.path.isdir(model_path):
            logger.info(f"{model_path} is a directory")
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, config=config, device_map="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            logger.info(f"{model_path} is not a directory, assuming it's a Hugging Face model ID")
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, config=config, device_map="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Set pad token and chat template
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
        
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_evaluation_dataset(dataset_path):
    #logger.info(f"Loading evaluation dataset from {dataset_path}")
    return load_dataset("json", data_files=os.path.join(dataset_path, "test.json"), split="train")

def evaluate_model(model, tokenizer, dataset, max_new_tokens=50,num_samples=10):
    logger.info("Starting model evaluation")
    try:
        # Check CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Move model to the appropriate device
        #model = model.to(device)
        model.eval()

        predictions = []
        references = []


        # Create a text-generation pipeline
        logger.info("Creating text generation pipeline")
        text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

        # Take only a subset of the dataset for quicker testing
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        logger.info(f"Evaluating on {len(dataset)} samples")

        for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
            if i % 10 == 0:
                logger.info(f"Processing example {i+1}/{len(dataset)}")

            question = example['messages'][0]['content']
            context = example['messages'][1]['content']
            reference = example['messages'][2]['content']

            prompt = f"Human: Answer the following question based on the given context.\n\nContext: {context}\n\nQuestion: {question}\n\nAssistant:"

            try:
                output = text_generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
                prediction = output[0]['generated_text']
            except Exception as e:
                logger.error(f"Error generating text for example {i}: {e}")
                prediction = ""
            
            predictions.append(prediction)
            references.append(reference)

        logger.info("Evaluation completed")
        return predictions, references
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def compute_metrics(predictions, references):
    #logger.info("Computing metrics")
    # For simplicity, we'll use accuracy and F1 score
    # You might want to use more sophisticated metrics for QA tasks
    accuracy = accuracy_score([ref.lower() for ref in references], [pred.lower() for pred in predictions])
    f1 = f1_score([ref.lower() for ref in references], [pred.lower() for pred in predictions], average='weighted')
    return {"accuracy": accuracy, "f1_score": f1}

def main(model_path, test_file, output_dir, num_samples):
    # Install requirements
    #/opt/ml/processing/input/code
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    install_requirements(requirements_file)

    model, tokenizer = load_model_and_tokenizer(model_path)
    eval_dataset = load_evaluation_dataset(test_file)

    predictions, references = evaluate_model(model, tokenizer, eval_dataset, num_samples=num_samples)
    metrics = compute_metrics(predictions, references)

    #logger.info(f"Evaluation metrics: {metrics}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(metrics, f)

    #logger.info(f"Evaluation results saved to {args.output_dir}/evaluation_results.json")

if __name__ == "__main__":
    model_path='/opt/ml/processing/input/model'
    test_file='/opt/ml/processing/input/data'
    output_dir='/opt/ml/processing/output'
    num_samples=10
    print_library_versions()
    main(model_path, test_file, output_dir, num_samples)