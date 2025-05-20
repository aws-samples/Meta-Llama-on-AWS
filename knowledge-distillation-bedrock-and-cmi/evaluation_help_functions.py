# Standard Library Imports
import concurrent.futures
import json
import jsonlines
import logging
import os
import random
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime
import math

import matplotlib.pyplot as plt
import numpy as np
import csv
from IPython.display import Markdown, display

# Third-Party Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from tqdm import tqdm
import numpy as np
from IPython.display import Markdown, display


# AWS Imports
import boto3
from botocore.exceptions import ClientError
from sagemaker.s3 import S3Uploader

os.makedirs('logs', exist_ok=True)



def create_payload(
    prompt: str,
    system_message: str = None,
    parameters: dict = {
        "max_gen_len": 512,
        "temperature": 0.0,
        "top_p": 0.9
    }
) -> dict:
    """
    Creates a payload for Llama model invocation using the instruct format.
    
    Args:
        prompt (str): The main prompt/question for the model
        system_message (str, optional): System message to set context/behavior
        parameters (dict): Model parameters like max_tokens_to_sample, temperature, etc.
    
    Returns:
        dict: Formatted payload for model invocation
    """
    if not prompt:
        raise ValueError("Please provide a non-empty prompt.")
    
    # Construct the prompt format using Llama style
    if system_message:
        prompt_data = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        prompt_data = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
    # Merge the prompt with allowed parameters
    payload = {
        "prompt": prompt_data,
        "max_gen_len": parameters.get("max_gen_len", 250),
        "temperature": parameters.get("temperature", 0.0),
        "top_p": parameters.get("top_p", 0.9)
    }
    
    return json.dumps(payload)

# Create a more robust logging setup
def setup_logging(timestamp):
    """Set up logging with both file and console handlers"""
    # Create logs directory in current working directory
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename
    log_file = os.path.join(log_dir, f'model_comparison_{timestamp}.log')
    
    # Remove any existing handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Configure logging
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return log_file

class ModelMetrics:
    def __init__(self):
        self.request_times = []
        self.input_tokens = []
        self.output_tokens = []
        self.start_times = {}
        self.active_hours = set()
        self.request_count = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_start = time.time()  # Add total processing time tracking

    def start_request(self):
        """Start timing a request"""
        self.start_times[threading.get_ident()] = time.time()
        self.active_hours.add(time.localtime().tm_hour)

    def record_request(self, input_text, output_text, success=True):
        """Record metrics for a completed request"""
        thread_id = threading.get_ident()
        if thread_id in self.start_times:
            duration = time.time() - self.start_times[thread_id]
            del self.start_times[thread_id]
        else:
            duration = 0

        # Simple token estimation (approximate)
        input_tokens = len(input_text.split())
        output_tokens = len(output_text.split()) if output_text else 0
        
        self.request_times.append(duration)
        self.input_tokens.append(input_tokens)
        self.output_tokens.append(output_tokens)
        self.request_count += 1
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            logger = logging.getLogger()
            logger.error(f"Failed request - Input: {input_text[:100]}... Output length: {len(output_text)}")

    def calculate_metrics(self):
        """Calculate all metrics"""
        if not self.request_times:
            return {}
            
        total_processing_time = time.time() - self.total_processing_start
        total_time_mins = total_processing_time / 60
        total_input_tokens = sum(self.input_tokens)
        total_output_tokens = sum(self.output_tokens)

        metrics = {
            'total_processing_time_seconds': total_processing_time,
            'total_processing_time_minutes': total_time_mins,
            'peak_input_tpm': max(self.input_tokens) * (60 / min([t for t in self.request_times if t > 0] or [1])),
            'peak_output_tpm': max(self.output_tokens) * (60 / min([t for t in self.request_times if t > 0] or [1])),
            'peak_load_hours': len(self.active_hours),
            'avg_input_tpm': total_input_tokens / total_time_mins if total_time_mins > 0 else 0,
            'avg_output_tpm': total_output_tokens / total_time_mins if total_time_mins > 0 else 0,
            'avg_load_hours': len(self.active_hours) / 24,
            'avg_rpm': self.request_count / total_time_mins if total_time_mins > 0 else 0,
            'avg_input_tokens_per_request': total_input_tokens / self.request_count if self.request_count > 0 else 0,
            'avg_output_tokens_per_request': total_output_tokens / self.request_count if self.request_count > 0 else 0,
            'avg_latency': sum(self.request_times) / len(self.request_times) if self.request_times else 0,
            'max_latency': max(self.request_times) if self.request_times else 0,
            'min_latency': min(t for t in self.request_times if t > 0) if any(t > 0 for t in self.request_times) else 0,
            'total_requests': self.request_count,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.successful_requests / self.request_count * 100) if self.request_count > 0 else 0
        }
        return metrics


# Set up logging with immediate flush
class ImmediateLogger(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/model_comparison_{time.strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.addHandler(ImmediateLogger(sys.stdout))

class ProgressStats:
    def __init__(self):
        self.total = 0
        self.errors = Counter()
        self.success = 0
        self.failed_requests = defaultdict(list)
        self.skipped = 0

def process_line(line, model_id, model_name, bedrock_runtime, metrics):
    logger = logging.getLogger()
    
    if "question" not in line or not line["question"].strip():
        logger.warning(f"Skipped empty or invalid question for model {model_name}")
        return None, "skipped", None

    question = line["question"]
    reference = line.get("answers", "")
    
    try:
        metrics.start_request()
        
        payload = create_payload(
            prompt=question,
            system_message="You are an AI assistant helping to answer questions about biomedical research literature.",
            parameters={
                "max_gen_len": 512,
                "temperature": 0.0,
                "top_p": 0.9
            }
        )
        
        try:
            response = bedrock_runtime.invoke_model(
                body=payload,
                modelId=model_id,
                accept="application/json",
                contentType="application/json"
            )
            
            response_body = json.loads(response.get("body").read())
            logger.debug(f"Model {model_name} - Raw response: {response_body}")
            
            # Extract model output
            if 'generation' in response_body:
                model_output = response_body['generation'].strip()
                if model_output:
                    metrics.record_request(question, model_output, success=True)
                    
                    # Format for Bedrock evaluation with full model identifier
                    output_format = {
                        "prompt": question,
                        "referenceResponse": reference if reference else None,
                        "category": "Biomedical Literature",
                        "modelResponses": [
                            {
                                "response": model_output,
                                "modelIdentifier": model_name  # This will now be the full model name from config
                            }
                        ]
                    }
                    return output_format, "success", None
            
            logger.error(f"Model {model_name} - Invalid or empty response format")
            metrics.record_request(question, "", success=False)
            return None, "empty_response", {"question": question, "original_line": line}
                
        except ClientError as ce:
            error_code = ce.response['Error']['Code']
            error_message = ce.response['Error']['Message']
            logger.error(f"Model {model_name} - ClientError: {error_code} - {error_message}")
            metrics.record_request(question, "", success=False)
            return None, "client_error", {"question": question, "original_line": line}
            
    except Exception as e:
        logger.error(f"Model {model_name} - Unexpected error: {str(e)}")
        logger.error("Full error details:", exc_info=True)
        metrics.record_request(question, "", success=False)
        return None, "unexpected_error", {"question": question, "original_line": line}
    
def process_file(input_file, output_file, model_id, model_name, bedrock_runtime, max_workers=5):
    stats = ProgressStats()
    metrics = ModelMetrics()
    processed_results = []
    
    with jsonlines.open(input_file) as input_fh, jsonlines.open(output_file, mode='w') as output_fh:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for line in input_fh:
                if "question" not in line or not line["question"].strip():
                    stats.skipped += 1
                    continue
                future = executor.submit(process_line, line, model_id, model_name, bedrock_runtime, metrics)
                futures.append(future)
            
            with tqdm(total=len(futures), desc="Processing", file=sys.stdout) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result, status, failed_data = future.result()
                        stats.total += 1
                        
                        if status == "success" and result:
                            stats.success += 1
                            output_fh.write(result)  # Write the properly formatted result
                            processed_results.append(result)
                        elif status == "skipped":
                            stats.skipped += 1
                        else:
                            stats.errors[status] += 1
                            if failed_data:
                                logger.error(f"Failed request: {failed_data}")
                        
                        pbar.set_description(
                            f"Processed: {stats.total} | "
                            f"Success: {stats.success} | "
                            f"Failed: {sum(stats.errors.values())} | "
                            f"Skipped: {stats.skipped}"
                        )
                        pbar.refresh()
                        
                    except Exception as e:
                        stats.errors["unexpected_error"] += 1
                        logger.error(f"Unexpected error in future processing: {str(e)}")
                    
                    pbar.update(1)
                    sys.stdout.flush()

    return {
        'metrics': metrics.calculate_metrics(),
        'processed_results': processed_results,
        'stats': {
            'total': stats.total,
            'success': stats.success,
            'skipped': stats.skipped,
            'errors': dict(stats.errors)
        }
    }

# Modified analyze_errors function
def analyze_errors(log_file):
    """Analyze errors from the log file and print a summary."""
    if not os.path.exists(log_file):
        print(f"Warning: Log file not found at {log_file}")
        return defaultdict(int), defaultdict(list)
        
    error_counts = defaultdict(int)
    error_examples = defaultdict(list)
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'ERROR' in line:
                    for error_type in ['ClientError', 'empty_response', 'unexpected_error']:
                        if error_type in line:
                            error_counts[error_type] += 1
                            if len(error_examples[error_type]) < 3:  # Keep up to 3 examples
                                error_examples[error_type].append(line.strip())
    except Exception as e:
        print(f"Error reading log file: {str(e)}")
        return defaultdict(int), defaultdict(list)
    
    print("\nError Analysis:")
    print("=" * 50)
    for error_type, count in error_counts.items():
        print(f"\n{error_type}: {count} occurrences")
        print("Example errors:")
        for example in error_examples[error_type]:
            print(f"  - {example}")
            
    return error_counts, error_examples


def retry_failed_requests(retry_file, output_file, model_id, bedrock_runtime, max_workers=2):
    """Process a file of failed requests with reduced concurrency and increased delays."""
    print(f"\nProcessing retry file: {retry_file}")
    time.sleep(5)  # Add initial delay before starting retries
    process_file(retry_file, f"retry_results_{time.strftime('%Y%m%d_%H%M%S')}.jsonl", 
                model_id, bedrock_runtime, max_workers=max_workers)
    

# Usage example:
def run_model_comparison(input_file, model_configs,bedrock_runtime, sample_size=None, timestamp=None):
    """
    Run comparison across multiple models with custom naming
    
    Args:
        input_file (str): Input file path
        model_configs (dict): Dictionary containing model configurations
        sample_size (int, optional): Number of samples to process (None for all)
        timestamp (str, optional): Custom timestamp for file naming
    """
    if timestamp is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Load and sample data if needed
    with jsonlines.open(input_file) as reader:
        all_data = list(reader)
    
    if sample_size and sample_size < len(all_data):
        sampled_data = random.sample(all_data, sample_size)
        temp_input_file = f"temp_sample_{timestamp}.jsonl"
        with jsonlines.open(temp_input_file, mode='w') as writer:
            writer.write_all(sampled_data)
        input_file_to_use = temp_input_file
        print(f"\nUsing {sample_size} samples from the dataset")
    else:
        input_file_to_use = input_file
    
    results = {}
    try:
        for model_key, config in model_configs.items():
            print(f"\nProcessing with model: {config['model_name']}")  # Use the full model name
            output_file = f"{config['output_prefix']}_{timestamp}.jsonl"
            result = process_file(
                input_file_to_use, 
                output_file, 
                config['model_id'],
                config['model_name'],  # Pass the full model name
                bedrock_runtime
            )
            results[model_key] = {
                'model_id': config['model_id'],
                'model_name': config['model_name'],
                'output_file': output_file,
                **result
            }
    finally:
        # Clean up temporary file if it was created
        if 'temp_input_file' in locals():
            try:
                os.remove(temp_input_file)
            except:
                pass
                
    return results  



def create_radar_plot(data):
    # Extract metrics for both models
    models_data = {}
    for model_type, model_info in data.items():
        models_data[model_info['model_name']] = model_info['metrics']
    
    # Convert to DataFrame
    df = pd.DataFrame(models_data)
    
    # Define metrics where lower values are better
    metrics_lower_better = [
        'total_processing_time_seconds',
        'total_processing_time_minutes',
        'avg_latency',
        'max_latency',
        'min_latency'
    ]
    
    # Select metrics to display (you can adjust this list)
    metrics_to_display = [
        'avg_latency',
        'avg_input_tpm',
        'avg_output_tpm',
        'avg_rpm',
        'success_rate'
    ]
    
    # Filter DataFrame to include only selected metrics
    df_display = df.loc[metrics_to_display]
    
    # Normalize the data (0-1 scale)
    df_normalized = df_display.copy()
    for metric in df_normalized.index:
        if metric in metrics_lower_better:
            df_normalized.loc[metric] = df_display.loc[metric].min() / df_display.loc[metric]
        else:import pandas as pd # type: ignore

def create_dual_radar_plots(data):
    # Extract metrics for both models
    models_data = {}
    for model_type, model_info in data.items():
        models_data[model_info['model_name']] = model_info['metrics']
    
    # Convert to DataFrame
    df = pd.DataFrame(models_data)
    
    # Define metrics categories
    metrics_lower_better = [
        'total_processing_time_seconds',
        'total_processing_time_minutes',
        'avg_latency',
        'max_latency',
        'min_latency'
    ]
    
    metrics_higher_better = [
        'avg_input_tpm',
        'avg_output_tpm',
        'avg_rpm',
        'success_rate',
        'peak_input_tpm',
        'peak_output_tpm'
    ]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw=dict(projection='polar'))
    
    # Function to create radar plot
    def plot_radar(metrics, ax, title):
        df_display = df.loc[metrics]
        
        # Normalize the data (0-1 scale)
        df_normalized = df_display.copy()
        for metric in df_normalized.index:
            if metric in metrics_lower_better:
                df_normalized.loc[metric] = df_display.loc[metric].min() / df_display.loc[metric]
            else:
                df_normalized.loc[metric] = df_display.loc[metric] / df_display.loc[metric].max()
        
        # Set up the angles for the radar plot
        num_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot data for both models
        for idx, column in enumerate(df_normalized.columns):
            values = df_normalized[column].tolist()
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=column)
            ax.fill(angles, values, alpha=0.25)
        
        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics], size=8)
        
        # Add title
        ax.set_title(title, size=14, pad=20)
        
        return df_display
    
    # Create both plots
    df1 = plot_radar(metrics_lower_better, ax1, "Metrics where Lower is Better")
    df2 = plot_radar(metrics_higher_better, ax2, "Metrics where Higher is Better")
    
    # Add legend (only once, since it's the same for both plots)
    plt.figlegend(loc='center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    # Add explanatory note
    note = (
        "Note: These radar plots compare model performance across metrics.\n"
        "Left plot: Metrics where lower values are better (values are inverted so outward points show better performance)\n"
        "Right plot: Metrics where higher values are better (outward points directly show better performance)\n"
        "Larger area indicates better overall performance in both plots."
    )
    plt.figtext(0.02, 0.02, note, fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    return pd.concat([df1, df2]), fig

#CORS Configuration
def configure_bucket_cors(bucket_name):
    """
    Configure CORS for the S3 bucket used in model evaluation
    
    Args:
        bucket_name (str): Name of the S3 bucket
    """
    try:
        s3_client = boto3.client('s3')
        
        # Define the required CORS configuration
        cors_configuration = {
            'CORSRules': [
                {
                    'AllowedHeaders': ['*'],
                    'AllowedMethods': ['GET', 'PUT', 'POST', 'DELETE'],
                    'AllowedOrigins': ['*'],
                    'ExposeHeaders': ['Access-Control-Allow-Origin']
                }
            ]
        }
        
        # Apply CORS configuration
        try:
            s3_client.put_bucket_cors(
                Bucket=bucket_name,
                CORSConfiguration=cors_configuration
            )
            print(f"Successfully configured CORS for bucket: {bucket_name}")
            
            # Verify CORS configuration
            response = s3_client.get_bucket_cors(Bucket=bucket_name)
            print("\nVerified CORS Configuration:")
            print(json.dumps(response, indent=2))
            
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchCORSConfiguration':
                print(f"No existing CORS configuration found. Setting new configuration.")
                s3_client.put_bucket_cors(
                    Bucket=bucket_name,
                    CORSConfiguration=cors_configuration
                )
                print(f"Successfully configured CORS for bucket: {bucket_name}")
            else:
                raise
                
    except Exception as e:
        print(f"Error configuring CORS for bucket {bucket_name}: {str(e)}")
        raise

#File Upload Functions
#Upload Evaluation Files
def upload_evaluation_files_to_s3(model_configs, bucket,sess,timestamp=None):
    """
    Upload evaluation JSONL files to S3 bucket with clean paths
    
    Args:
        model_configs (dict): Dictionary containing model configurations
        timestamp (str, optional): Timestamp for file naming
        
    Returns:
        dict: Dictionary of model names and their S3 locations
    """
    if timestamp is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Create S3 path for evaluation
    s3_prefix = f"model-evaluation/{timestamp}"
    s3_base_path = f"s3://{bucket}/{s3_prefix}"
    
    print(f"Uploading files to base path: {s3_base_path}")
    
    # Dictionary to store S3 locations
    s3_locations = {}
    
    try:
        # Upload each model's output file
        for model_name, config in model_configs.items():
            local_file = f"{config['output_prefix']}_{timestamp}.jsonl"
            
            if os.path.exists(local_file):
                print(f"Found local file: {local_file}")
                
                # Create the S3 destination path
                s3_destination = f"{s3_base_path}/{model_name}"
                
                try:
                    # Upload and get the actual S3 location
                    actual_s3_location = S3Uploader.upload(
                        local_path=local_file,
                        desired_s3_uri=s3_destination,
                        sagemaker_session=sess
                    )
                    s3_locations[model_name] = actual_s3_location
                    print(f"Successfully uploaded {local_file} to {actual_s3_location}")
                except Exception as upload_error:
                    print(f"Error uploading {local_file}: {str(upload_error)}")
                    raise
            else:
                print(f"Warning: File {local_file} not found")
        
        # Upload comparison results if they exist
        comparison_file = f"model_comparison_{timestamp}.json"
        if os.path.exists(comparison_file):
            print(f"Found comparison file: {comparison_file}")
            
            # Create the S3 destination path for comparison file
            comparison_s3_destination = f"{s3_base_path}/comparison"
            
            try:
                # Upload and get the actual S3 location
                actual_comparison_location = S3Uploader.upload(
                    local_path=comparison_file,
                    desired_s3_uri=comparison_s3_destination,
                    sagemaker_session=sess
                )
                s3_locations['comparison'] = actual_comparison_location
                print(f"Successfully uploaded comparison results to {actual_comparison_location}")
            except Exception as upload_error:
                print(f"Error uploading comparison file: {str(upload_error)}")
                raise
        
        return s3_locations
        
    except Exception as e:
        print(f"Error in upload process: {str(e)}")
        raise

#Preparation and Verification
def prepare_and_upload_evaluation_files(model_configs, bucket,sess,timestamp=None):
    """
    Prepare bucket and upload evaluation files
    """
    if timestamp is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    try:
        # Configure CORS for the bucket
        print("\nConfiguring CORS for S3 bucket...")
        configure_bucket_cors(bucket)
        
        # Upload files
        print("\nUploading evaluation files...")
        s3_locations = upload_and_verify_files(model_configs,bucket,sess, timestamp)
        
        return s3_locations
        
    except Exception as e:
        print(f"Error in preparation and upload process: {str(e)}")
        raise




#Upload Verification
def upload_and_verify_files(model_configs,bucket,sess,timestamp=None):
    """Upload files and verify they exist in S3"""
    if timestamp is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    try:
        # Upload files
        s3_locations = upload_evaluation_files_to_s3(model_configs,bucket,sess, timestamp)
        
        # Verify uploads using boto3
        s3_client = boto3.client('s3')
        
        print("\nVerifying uploaded files:")
        for model_name, s3_uri in s3_locations.items():
            # Parse S3 URI
            s3_path = s3_uri.replace('s3://', '').split('/')
            bucket_name = s3_path[0]
            key = '/'.join(s3_path[1:])
            
            try:
                # Check if file exists
                s3_client.head_object(Bucket=bucket_name, Key=key)
                print(f"✓ Verified {model_name}: {s3_uri}")
            except Exception as e:
                print(f"✗ Failed to verify {model_name}: {s3_uri}")
                print(f"Error: {str(e)}")
        
        return s3_locations
        
    except Exception as e:
        print(f"Error in upload and verify process: {str(e)}")
        raise


def get_bucket_cors_config(bucket_name):
    """Retrieve the CORS configuration rules of an Amazon S3 bucket

    :param bucket_name: string
    :return: List of the bucket's CORS configuration rules. If no CORS
    configuration exists, return empty list. If error, return None.
    """

    # Retrieve the CORS configuration
    s3 = boto3.client('s3')
    try:
        response = s3.get_bucket_cors(Bucket=bucket_name)
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchCORSConfiguration':
            return []
        else:
            # AllAccessDisabled error == bucket not found
            logging.error(e)
            return None
    return response['CORSRules']


def create_llm_judge_evaluation(
    client,
    job_name: str,
    role_arn: str,
    input_s3_uri: str,
    output_s3_uri: str,
    evaluator_model_id: str,
    inference_Source_Id: str,
    dataset_name: str = None,
    task_type: str = "General", # must be General for LLMaaJ,
    
):    
    # All available LLM-as-judge metrics
    llm_judge_metrics = [
        "Builtin.Correctness",
        "Builtin.Completeness", 
        "Builtin.Faithfulness",
        "Builtin.Helpfulness",
        "Builtin.Coherence",
        "Builtin.Relevance",
        "Builtin.FollowingInstructions",
        "Builtin.ProfessionalStyleAndTone",
        "Builtin.Harmfulness",
        "Builtin.Stereotyping",
        "Builtin.Refusal"
    ]

    # Configure dataset
    dataset_config = {
        "name": dataset_name or "CustomDataset",
        "datasetLocation": {
            "s3Uri": input_s3_uri
        }
    }

    try:
        response = client.create_evaluation_job(
            jobName=job_name,
            roleArn=role_arn,
            applicationType="ModelEvaluation",
            evaluationConfig={
                "automated": {
                    "datasetMetricConfigs": [
                        {
                            "taskType": task_type,
                            "dataset": dataset_config,
                            "metricNames": llm_judge_metrics
                        }
                    ],
                    "evaluatorModelConfig": {
                        "bedrockEvaluatorModels": [
                            {
                                "modelIdentifier": evaluator_model_id
                            }
                        ]
                    }
                }
            },
            inferenceConfig={
                "models": [
                    {
                        "precomputedInferenceSource": {
                            "inferenceSourceIdentifier": inference_Source_Id
                        }
                    }
                ]
            },
            outputDataConfig={
                "s3Uri": output_s3_uri
            }
        )
        return response
        
    except Exception as e:
        print(f"Error creating evaluation job: {str(e)}")
        raise



def construct_evaluation_key(eval_dict, model_type):
    """
    Constructs the S3 key prefix for the evaluation output file.
    """
    model_info = eval_dict[model_type]
    
    # Remove 's3://<bucket-name>/' from the output_location and any trailing slashes
    base_path = model_info['output_location'].split('/', 3)[-1].rstrip('/')
    
    # Add job name and job ID
    path = f"{base_path}/{model_info['job_name']}/{model_info['job_arn'].split('/')[-1]}"
    
    # Add model name
    path = f"{path}/models/{model_info['model_name']}"
    
    # Add the fixed parts
    path = f"{path}/taskTypes/General/datasets/CustomDataset"
    
    return path

def find_latest_evaluation_file(s3_client, bucket_name, prefix):
    """
    Lists objects in the S3 bucket with the given prefix and finds the most recent .jsonl file.
    """
    latest_file = None
    latest_time = None
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('_output.jsonl'):
                        if latest_time is None or obj['LastModified'] > latest_time:
                            latest_time = obj['LastModified']
                            latest_file = obj['Key']
        return latest_file
    except Exception as e:
        print(f"Error accessing S3: {e}")
        return None

def get_evaluation_files(eval_dict, bucket):
    # Convert S3.Bucket object to string bucket name
    bucket_name = bucket.name if hasattr(bucket, 'name') else str(bucket)
    s3_client = boto3.client('s3')
    
    results = {}
    for model_type in ['base', 'student']:
        prefix = construct_evaluation_key(eval_dict, model_type)
        print(f"Searching with prefix: {prefix}")  # Debug print
        file_key = find_latest_evaluation_file(s3_client, bucket_name, prefix)
        results[model_type] = file_key
    
    return results

def read_and_organize_metrics_from_s3(bucket_name, file_key):
    s3_client = boto3.client('s3')
    metrics_dict = {}
    
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        
        for line in content.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    if 'automatedEvaluationResult' in data and 'scores' in data['automatedEvaluationResult']:
                        for score in data['automatedEvaluationResult']['scores']:
                            metric_name = score.get('metricName')
                            if not metric_name:
                                continue
                                
                            metric_value = score.get('result')
                            if metric_value is None:
                                continue
                                
                            # Convert to float if possible
                            try:
                                metric_value = float(metric_value)
                            except (TypeError, ValueError):
                                print(f"Warning: Invalid metric value for {metric_name}: {metric_value}")
                                continue
                                
                            if metric_name not in metrics_dict:
                                metrics_dict[metric_name] = []
                            metrics_dict[metric_name].append(metric_value)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {e}")
                    continue
        
        # Calculate mean for each metric, filtering out None values
        mean_metrics = {}
        for metric, values in metrics_dict.items():
            valid_values = [v for v in values if v is not None]
            if valid_values:
                mean_metrics[metric] = np.mean(valid_values)
            else:
                print(f"Warning: No valid values for metric {metric}")
                
        if not mean_metrics:
            print(f"No valid metrics found in file: {file_key}")
            return None
            
        return mean_metrics
    
    except Exception as e:
        print(f"Error reading file {file_key}: {e}")
        return None

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes."""
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, **kwargs):
            return super().fill_between(*args, **kwargs)

        def plot(self, *args, **kwargs):
            return super().plot(*args, **kwargs)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

    register_projection(RadarAxes)
    return theta

def plot_radar_chart(metrics1, metrics2, title="Model Comparison Radar Chart"):
    # Get common metrics
    common_metrics = list(set(metrics1.keys()) & set(metrics2.keys()))
    
    if not common_metrics:
        print("No common metrics found between the models")
        return None
    
    # Normalize metric names for better display
    metric_display_names = [m.replace("Builtin.", "") for m in common_metrics]
    
    # Get the values for each model
    model1_values = []
    model2_values = []
    final_metrics = []
    
    for metric in common_metrics:
        val1 = metrics1.get(metric)
        val2 = metrics2.get(metric)
        
        if val1 is not None and val2 is not None:
            model1_values.append(val1)
            model2_values.append(val2)
            final_metrics.append(metric)
    
    if not model1_values or not model2_values:
        print("No valid comparison metrics found")
        return None
    
    # Set up the radar chart
    num_vars = len(final_metrics)
    theta = radar_factory(num_vars, frame='polygon')
    
    # Create the figure and subplot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))
    
    # Plot the data
    ax.plot(theta, model1_values, 'o-', label='Base Model')
    ax.fill(theta, model1_values, alpha=0.25)
    ax.plot(theta, model2_values, 'o-', label='Fine-tuned Model')
    ax.fill(theta, model2_values, alpha=0.25)
    
    # Set chart properties
    ax.set_varlabels([m.replace("Builtin.", "") for m in final_metrics])
    ax.set_title(title, pad=20)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    return fig

def create_comparison_table(metrics1, metrics2):
    """Create a formatted comparison table of metrics with improvement indicators"""
    common_metrics = set(metrics1.keys()) & set(metrics2.keys())
    
    if not common_metrics:
        return "No common metrics found between the models"
    
    # Calculate maximum lengths for formatting
    metric_col_width = max(len("Metric"), max(len(m) for m in common_metrics))
    score_col_width = max(len("Base Model"), len("Fine-tuned Model"), 12)
    diff_col_width = max(len("Difference"), 15)  # Increased to accommodate arrows
    
    # Create horizontal line
    horizontal_line = f"+{'-' * (metric_col_width + 2)}+{'-' * (score_col_width + 2)}+{'-' * (score_col_width + 2)}+{'-' * (diff_col_width + 2)}+"
    
    # Create header
    header = (f"| {'Metric':<{metric_col_width}} "
             f"| {'Base Model':<{score_col_width}} "
             f"| {'Fine-tuned Model':<{score_col_width}} "
             f"| {'Difference':<{diff_col_width}} |")
    
    # Create table rows
    rows = []
    for metric in sorted(common_metrics):
        val1 = metrics1.get(metric)
        val2 = metrics2.get(metric)
        if val1 is not None and val2 is not None:
            diff = val2 - val1  # Changed to val2 - val1 to show improvement of fine-tuned model
            metric_name = metric.replace("Builtin.", "")
            
            # Add arrow indicating improvement/decline
            if diff > 0:
                arrow = "↑"  # Improvement
            elif diff < 0:
                arrow = "↓"  # Decline
            else:
                arrow = "="  # No change
                
            row = (f"| {metric_name:<{metric_col_width}} "
                  f"| {val1:>{score_col_width}.4f} "
                  f"| {val2:>{score_col_width}.4f} "
                  f"| {diff:>{diff_col_width-2}.4f} {arrow} |")
            rows.append(row)
    
    # Combine all parts of the table
    table = "\n".join([
        horizontal_line,
        header,
        horizontal_line,
        "\n".join(rows),
        horizontal_line
    ])
    
    return table

    
def create_table_image(metrics1, metrics2, output_file="metrics_comparison_table.png"):
    """Create and save a table as an image using matplotlib with color-coded improvements"""
    common_metrics = set(metrics1.keys()) & set(metrics2.keys())
    
    if not common_metrics:
        return "No common metrics found between the models"
    
    # Prepare data for table
    table_data = []
    cell_colors = []
    arrows = []
    
    for metric in sorted(common_metrics):
        val1 = metrics1.get(metric)
        val2 = metrics2.get(metric)
        if val1 is not None and val2 is not None:
            diff = val2 - val1  # Changed to val2 - val1 to show improvement of fine-tuned model
            metric_name = metric.replace("Builtin.", "")
            
            # Add arrow indicating improvement/decline
            if diff > 0:
                arrow = "↑"  # Improvement
                row_color = ['#f2f2f2', '#f2f2f2', '#e6ffe6', '#e6ffe6']  # Light green for improvement
            elif diff < 0:
                arrow = "↓"  # Decline
                row_color = ['#f2f2f2', '#f2f2f2', '#ffe6e6', '#ffe6e6']  # Light red for decline
            else:
                arrow = "="  # No change
                row_color = ['#f2f2f2'] * 4  # Normal background
                
            table_data.append([
                metric_name, 
                f"{val1:.4f}", 
                f"{val2:.4f}", 
                f"{diff:.4f} {arrow}"
            ])
            cell_colors.append(row_color)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Metric', 'Base Model', 'Fine-tuned Model', 'Difference'],
                    cellLoc='center',
                    loc='center',
                    cellColours=cell_colors,
                    colColours=['#e6e6e6']*4)  # Header color
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Add a legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#e6ffe6', label='Improvement'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#ffe6e6', label='Decline'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.3))
    
    # Add title
    plt.title('Metrics Comparison (Base vs Fine-tuned Model)', pad=20)
    
    # Save the figure
    plt.savefig(output_file, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    return output_file

def analyze_and_plot_metrics(bucket_name, file_key1, file_key2):
    print("Reading metrics from files...")
    print(f"File 1: {file_key1}")
    metrics1 = read_and_organize_metrics_from_s3(bucket_name, file_key1)
    
    print(f"\nFile 2: {file_key2}")
    metrics2 = read_and_organize_metrics_from_s3(bucket_name, file_key2)
    
    if metrics1 is None or metrics2 is None:
        print("Failed to read metrics files")
        return
    
    # Fixed filename for CSV
    csv_filename = "metrics_comparison.csv"
    
    # Prepare data for CSV
    common_metrics = set(metrics1.keys()) & set(metrics2.keys())
    
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['Metric', 'Base Model', 'Fine-tuned Model', 'Difference', 'Change Direction'])
            
            # Write data rows
            for metric in sorted(common_metrics):
                val1 = metrics1.get(metric)
                val2 = metrics2.get(metric)
                if val1 is not None and val2 is not None:
                    diff = val2 - val1
                    metric_name = metric.replace("Builtin.", "")
                    
                    # Determine change direction
                    if diff > 0:
                        direction = "↑ Improvement"
                    elif diff < 0:
                        direction = "↓ Decline"
                    else:
                        direction = "= No Change"
                    
                    writer.writerow([
                        metric_name,
                        f"{val1:.4f}",
                        f"{val2:.4f}",
                        f"{diff:.4f}",
                        direction
                    ])
        
        print(f"\nCSV file created successfully: {csv_filename}")
    except Exception as e:
        print(f"Error creating CSV file: {str(e)}")
    
    # Print text table comparison
    print("\nMetrics Comparison Table (Text format):")
    print(create_comparison_table(metrics1, metrics2))
    
    # Create and save table as image
    print("\nGenerating table image...")
    table_image = create_table_image(metrics1, metrics2)
    print(f"Table image saved as: {table_image}")
    
    # Create and show radar plot
    print("\nGenerating radar plot...")
    fig = plot_radar_chart(metrics1, metrics2)
    
    if fig is not None:
        output_file = "model_comparison_radar.png"
        fig.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"\nRadar plot saved as: {output_file}")
        plt.close(fig)
    else:
        print("Failed to generate radar plot")


def generate_model_comparison_report(csv_file_path):
    """
    Generate a markdown report comparing model performance based on CSV data.
    
    Args:
        csv_file_path (str): Path to the CSV file containing comparison metrics
    
    Returns:
        str: Markdown formatted report
    """
    # Load the data
    df = pd.read_csv(csv_file_path)
    
    # Set the first column as index if it's not already
    if not df.index.name and len(df.columns) > 2:
        df = df.set_index(df.columns[0])
    
    # Extract model names
    model_names = df.columns.tolist()
    finetuned_model = model_names[0]
    base_model = model_names[1]
    
    # Calculate performance improvements
    improvements = {}
    for metric in df.index:
        fine_tuned_value = df.loc[metric, finetuned_model]
        base_value = df.loc[metric, base_model]
        
        # For latency metrics, lower is better (calculate percentage improvement)
        if 'latency' in metric.lower() or 'time' in metric.lower():
            improvement = (base_value - fine_tuned_value) / base_value * 100
            improvements[metric] = f"{improvement:.2f}% faster"
        # For throughput metrics, higher is better
        elif 'tpm' in metric.lower() or 'rpm' in metric.lower():
            improvement = (fine_tuned_value - base_value) / base_value * 100
            improvements[metric] = f"{improvement:.2f}% higher"
    
    # Create markdown report
    markdown_report = f"""


## Testing Methodology
- **Models Compared**: {finetuned_model} vs {base_model}
- **Dataset**: Biomedical literature questions
- **Evaluation criteria**: Latency, throughput, success rate
## Performance Comparison

### Latency Metrics
| Metric | {finetuned_model} | {base_model} | Improvement |
|--------|------------|-----------|-------------|
| Avg Latency | {df.loc['avg_latency', finetuned_model]:.2f}s | {df.loc['avg_latency', base_model]:.2f}s | {improvements['avg_latency']} |
| Max Latency | {df.loc['max_latency', finetuned_model]:.2f}s | {df.loc['max_latency', base_model]:.2f}s | {improvements['max_latency']} |
| Min Latency | {df.loc['min_latency', finetuned_model]:.2f}s | {df.loc['min_latency', base_model]:.2f}s | {improvements['min_latency']} |

### Throughput Metrics
| Metric | {finetuned_model} | {base_model} | Improvement |
|--------|------------|-----------|-------------|
| Input Tokens/min | {df.loc['avg_input_tpm', finetuned_model]:.2f} | {df.loc['avg_input_tpm', base_model]:.2f} | {improvements['avg_input_tpm']} |
| Output Tokens/min | {df.loc['avg_output_tpm', finetuned_model]:.2f} | {df.loc['avg_output_tpm', base_model]:.2f} | {improvements['avg_output_tpm']} |
| Requests/min | {df.loc['avg_rpm', finetuned_model]:.2f} | {df.loc['avg_rpm', base_model]:.2f} | {improvements['avg_rpm']} |
| Peak Input TPM | {df.loc['peak_input_tpm', finetuned_model]:.2f} | {df.loc['peak_input_tpm', base_model]:.2f} | {improvements['peak_input_tpm']} |
| Peak Output TPM | {df.loc['peak_output_tpm', finetuned_model]:.2f} | {df.loc['peak_output_tpm', base_model]:.2f} | {improvements['peak_output_tpm']} |

### Success Rate
| Metric | {finetuned_model} | {base_model} |
|--------|------------|-----------|
| Success Rate | {df.loc['success_rate', finetuned_model]:.2f}% | {df.loc['success_rate', base_model]:.2f}% |

## Visualization
![Model Comparison Radar Chart](model_comparison_radar_plots.png)
*Figure 1: Performance comparison across metrics*

## Key Insights

The fine-tuned model ({finetuned_model}) shows significantly better performance than the base Bedrock model ({base_model}):

- **Latency**: {improvements['avg_latency']} average response time
- **Throughput**: {improvements['avg_output_tpm']} output token processing rate
- **Efficiency**: {improvements['avg_rpm']} more requests processed per minute
### Custom Model Import Benefits

Custom model imports in AWS Bedrock offer several advantages over on-demand Bedrock models:

1. **Performance**: As demonstrated, custom fine-tuned models can significantly outperform base models for specialized tasks
2. **Cost-Efficiency**: Custom model imports often provide more favorable pricing for high-volume applications
3. **Specialization**: Fine-tuned models deliver better results for domain-specific use cases

### When to Use Custom Model Import

Consider using custom model imports when:
- You need domain-specific knowledge not available in base models
- Your application requires consistent, predictable pricing at scale
- You have specialized requirements that benefit from fine-tuning
- You want to optimize for latency in production environments
"""
    
    return markdown_report




def generate_model_comparison_report_knowledge(csv_file_path):
    """
    Generate a detailed markdown report based on model comparison metrics from a CSV file.
    
    Args:
        csv_file_path: Path to the CSV file with comparison metrics
        
    Returns:
        Markdown formatted report that can be displayed in a Jupyter notebook
    """
    # Load the comparison data
    df = pd.read_csv(csv_file_path)
    
    # Count improvements and declines
    improvements = df[df['Change Direction'].str.contains('Improvement')].shape[0]
    declines = df[df['Change Direction'].str.contains('Decline')].shape[0]
    
    # Calculate average difference
    avg_difference = df['Difference'].mean()
    
    # Create markdown table from dataframe
    md_table = df.to_markdown(index=False)
    
    # Generate the full report
    report = f"""
## Model Comparison: Base vs Fine-tuned Model

### Performance Analysis

This report presents a comparison between the base Bedrock Llama 1B model and a fine-tuned custom variant. The evaluation was performed using LLM-based automated judgments across multiple quality metrics.

#### Key Findings

- **Metrics showing improvement in fine-tuned model:** {improvements} metrics
- **Metrics showing decline in fine-tuned model:** {declines} metrics
- **Average metric change:** {avg_difference:.4f}
#### Detailed Metrics Comparison

{md_table}

#### Interpretation

The metrics show that the fine-tuned model performs differently across various dimensions compared to the base model. Specifically:

- **Improvements in:** {', '.join(df[df['Change Direction'].str.contains('Improvement')]['Metric'].tolist())}
- **Declines in:** {', '.join(df[df['Change Direction'].str.contains('Decline')]['Metric'].tolist())}

These results demonstrate that fine-tuning produced targeted improvements in specific areas while potentially trading off performance in others. This is typical in fine-tuning processes, which often require multiple iterations to optimize across all dimensions.
### Visualization
![Model Comparison Radar Chart](model_comparison_radar.png)
*Figure 1:  Comparison across metrics*

### Cost and Usage Considerations

#### Price Comparison
- **Bedrock Base Models:** Pay-per-use pricing with no infrastructure management
- **Custom Model Import:** 
  - No cost for importing custom weights
  - On-Demand pricing based on active model copies
  - Billed in 5-minute increments
  - Potential cost savings for high-volume applications

#### When to Use Custom Model Import
1. **Domain-Specific Applications:** For specialized knowledge and industry-specific vocabulary
2. **Enterprise Integration:** Seamless deployment of models from SageMaker, EC2, or on-premises
3. **Unified Development:** Access to same tools as base models (knowledge bases, guardrails, evaluation)
4. **Operational Efficiency:** Serverless deployment without infrastructure management
5. **Security and Compliance:** Enterprise-grade security controls and model governance

#### Model Architecture Support
- Supports importing model architectures like Llama 2, Llama3, Llama3.1, Llama3.2, and Llama 3.3, Flan and Mistral
- Compatible with Hugging Face safetensors format
- Integration with existing AWS services (SageMaker, S3)

### Optimization Process
Fine-tuning and customization is an iterative process that requires:
1. Training with different hyperparameters
2. Evaluation on target metrics
3. Error analysis and dataset refinement
4. RAG implementation for context optimization
5. Industry-specific vocabulary alignment

The results presented here represent a specific snapshot in this process and further optimization may yield improvements across more metrics.
"""

    # Return the markdown report that can be displayed with display(Markdown(report))
    return report

