import boto3
import json
import time
import os
import subprocess
import shutil
from botocore.exceptions import ClientError
import random
import pandas as pd
from typing import List, Dict
from datetime import datetime
from IPython.display import display, clear_output, HTML



def create_boto3_layer(lambda_client):
    """Create a Lambda layer with the latest boto3 version"""
    try:
        # Create directories
        os.makedirs('boto3-layer/python', exist_ok=True)

        # Install boto3 into the layer directory
        subprocess.check_call([
            'pip', 'install', 'boto3==1.35.16', '-q','-t', 'boto3-layer/python',
            '--upgrade', '--no-cache-dir'
        ])

        # Create zip file
        shutil.make_archive('boto3-layer', 'zip', 'boto3-layer')

        # Upload to AWS as a Lambda layer
        with open('boto3-layer.zip', 'rb') as zip_file:
            response = lambda_client.publish_layer_version(
                LayerName='boto3-latest',
                Description='Latest Boto3 layer',
                Content={
                    'ZipFile': zip_file.read()
                },
                CompatibleRuntimes=['python3.10', 'python3.11']
            )

        layer_version_arn = response['LayerVersionArn']
        print(f"Created Lambda layer: {layer_version_arn}")

        # Clean up
        shutil.rmtree('boto3-layer')
        os.remove('boto3-layer.zip')

        return layer_version_arn

    except Exception as e:
        print(f"Error creating Lambda layer: {str(e)}")
        raise e
    


def get_pipeline_status(execution):
    try:
        return execution.describe()['PipelineExecutionStatus']
    except ClientError as e:
        print(f"Error getting pipeline status: {e}")
        return None

def get_step_statuses(execution):
    try:
        steps = execution.list_steps()
        return {step['StepName']: step['StepStatus'] for step in steps}
    except ClientError as e:
        print(f"Error getting step statuses: {e}")
        return {}

def is_pipeline_finished(status):
    return status in ['Succeeded', 'Completed', 'Failed', 'Stopped']

def print_progress(status, step_statuses):
    print(f"\nPipeline status: {status}")
    print("Step statuses:")
    for step, status in step_statuses.items():
        print(f"  {step}: {status}")

def monitor_pipeline_execution(execution, check_interval=60):
    print("Pipeline execution started.")
    print("Status updates (checking every minute):")
    
    previous_step_statuses = {}
    start_time = time.time()
    
    while True:
        clear_output(wait=True)
        status = get_pipeline_status(execution)
        if status is None:
            print("Failed to get pipeline status. Retrying...")
            time.sleep(check_interval)
            continue

        step_statuses = get_step_statuses(execution)
        
        # Calculate progress
        completed_steps = sum(1 for s in step_statuses.values() if s in ['Succeeded', 'Failed', 'Stopped'])
        progress = int((completed_steps / len(step_statuses)) * 100)
        
        # Generate HTML output
        elapsed_time = int(time.time() - start_time)
        html_output = f"""
        <h3>Pipeline Execution Status: {status}</h3>
        <p>Elapsed Time: {elapsed_time // 60}m {elapsed_time % 60}s</p>
        <p>Progress: {progress}%</p>
        <div style="width:100%; background-color:#ddd;">
            <div style="width:{progress}%; height:20px; background-color:#4CAF50;"></div>
        </div>
        <h4>Step Statuses</h4>
        <ul>
        """
        
        for step, step_status in step_statuses.items():
            color = 'green' if step_status == 'Succeeded' else 'red' if step_status == 'Failed' else 'orange'
            html_output += f"<li><span style='color:{color};'>{step}: {step_status}</span></li>"
        
        html_output += "</ul>"
        
        display(HTML(html_output))
        print(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if is_pipeline_finished(status):
            break

        time.sleep(check_interval)

    print("\nPipeline execution finished.")


def wait_for_model_availability(model_name_filter, max_attempts=30, delay=60):
    """
    Check for the availability of a model in Bedrock, retrying until found or max attempts reached.
    
    :param model_name_filter: The name (or part of the name) of the model to look for
    :param max_attempts: Maximum number of attempts to check for the model
    :param delay: Delay in seconds between attempts
    :return: Model information if found, None otherwise
    """
    bedrock_client = boto3.client('bedrock')
    paginator = bedrock_client.get_paginator('list_imported_models')

    for attempt in range(max_attempts):
        try:
            response_iterator = paginator.paginate(
                nameContains=model_name_filter,
                sortBy='CreationTime',
                sortOrder='Descending'
            )

            for page in response_iterator:
                for model in page.get('modelSummaries', []):
                    if model_name_filter.lower() in model['modelName'].lower():
                        print(f"Model found on attempt {attempt + 1}:")
                        print(f"Model ARN: {model['modelArn']}")
                        print(f"Model Name: {model['modelName']}")
                        print(f"Creation Time: {model['creationTime']}")
                        return model  # Return the first matching model

            print(f"Attempt {attempt + 1}: Model not found. Retrying in {delay} seconds...")
            time.sleep(delay)

        except ClientError as e:
            print(f"An error occurred: {e}")
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)

    print(f"Model not found after {max_attempts} attempts.")
    return None

def load_conversations_from_s3(bucket_name: str, file_key: str, num_samples: int = None) -> List[Dict]:
    try:
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        conversations = []
        for line in response['Body'].iter_lines():
            if line.strip():
                try:
                    conversation = json.loads(line.decode('utf-8'))
                    conversations.append(conversation)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
        print(f"Loaded {len(conversations)} conversations from S3")
        if num_samples and num_samples < len(conversations):
            return random.sample(conversations, num_samples)
        return conversations
    except Exception as e:
        print(f"Error loading conversations from S3: {e}")
        return []

def create_html_table(df, num_examples=None):
    if df.empty:
        return "<p>No results to display</p>"
    
    if num_examples is None:
        num_examples = len(df)
    
    # Enhanced CSS styles
    styles = """
    <style>
        .container {
            max-width: 1200px;
            margin: 0 auto;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        }
        
        .title {
            color: #2c3e50;
            padding: 20px 0;
            font-size: 24px;
            font-weight: 600;
            border-bottom: 2px solid #eee;
            margin-bottom: 30px;
        }
        
        .results-table {
            width: 100%;
            margin: 20px 0;
            border-collapse: separate;
            border-spacing: 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .results-table th {
            background-color: #f8f9fa;
            padding: 15px 20px;
            text-align: left;
            font-weight: 600;
            color: #2c3e50;
            border-bottom: 2px solid #e9ecef;
            width: 150px;
        }
        
        .results-table td {
            padding: 15px 20px;
            border-bottom: 1px solid #e9ecef;
            vertical-align: top;
        }
        
        .example-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            margin: 30px 0 15px 0;
            border-radius: 8px;
            font-weight: 500;
            font-size: 18px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .token-info {
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 10px;
            padding: 8px 12px;
            background-color: #f8f9fa;
            border-radius: 4px;
            display: inline-block;
        }
        
        .question {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #4299e1;
            margin: 5px 0;
            font-size: 15px;
            line-height: 1.6;
        }
        
        .expected {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #48bb78;
            margin: 5px 0;
            font-size: 15px;
            line-height: 1.6;
        }
        
        .prediction {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #ed64a6;
            margin: 5px 0;
            font-size: 15px;
            line-height: 1.6;
        }
        
        .section-label {
            font-size: 12px;
            text-transform: uppercase;
            color: #6c757d;
            margin-bottom: 5px;
            letter-spacing: 0.5px;
        }
        
        .metrics-container {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        
        .metric {
            background-color: #f8f9fa;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.9em;
            color: #6c757d;
        }
        
        @media (max-width: 768px) {
            .results-table th {
                width: 100px;
            }
        }
    </style>
    """
    
    html = styles + '<div class="container">'
    html += '<div class="title">🤖 Model Evaluation Results</div>'
    
    for idx, row in df.head(num_examples).iterrows():
        html += f"""
        <div class="example-header">📝 Example #{idx+1}</div>
        <table class="results-table">
            <tr>
                <th>Question</th>
                <td>
                    <div class="section-label">USER INPUT</div>
                    <div class="question">{row['Question']}</div>
                </td>
            </tr>
            <tr>
                <th>Expected Answer</th>
                <td>
                    <div class="section-label">GROUND TRUTH</div>
                    <div class="expected">{row['Expected Answer']}</div>
                </td>
            </tr>
            <tr>
                <th>Model Prediction</th>
                <td>
                    <div class="section-label">MODEL OUTPUT</div>
                    <div class="prediction">{row['Model Prediction']}</div>
                    <div class="metrics-container">
                        <div class="metric">
                            📥 Input Tokens: {row['Input Tokens']}
                        </div>
                        <div class="metric">
                            📤 Output Tokens: {row['Output Tokens']}
                        </div>
                    </div>
                </td>
            </tr>
        </table>
        """
    
    html += '</div>'
    return html

def display_results(df, num_examples=None):
    html_content = create_html_table(df, num_examples)
    display(HTML(html_content))

def run_tests_from_s3(model_id: str, s3_uri: str, num_samples: int = 10, batch_size: int = 5):
    try:
        print(f"Starting test with S3 URI: {s3_uri}")
        s3_path = s3_uri.replace('s3://', '')
        bucket_name, *prefix_parts = s3_path.split('/')
        prefix = '/'.join(prefix_parts)
        
        conversations = load_conversations_from_s3(bucket_name, f"{prefix}/test.json", num_samples)
        if not conversations:
            print("No conversations loaded!")
            return pd.DataFrame()
        
        br_runtime = boto3.client('bedrock-runtime')
        results = []
        
        for batch_start in range(0, len(conversations), batch_size):
            batch_end = min(batch_start + batch_size, len(conversations))
            batch = conversations[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//batch_size + 1}/{-(-len(conversations)//batch_size)}")
            
            for idx, conversation in enumerate(batch, batch_start + 1):
                try:
                    messages = conversation.get('messages', [])
                    messages_by_role = {
                        'system': next((msg['content'] for msg in messages if msg['role'] == 'system'), None),
                        'user': [msg for msg in messages if msg['role'] == 'user'],
                        'assistant': [msg for msg in messages if msg['role'] == 'assistant']
                    }
                    
                    if not messages_by_role['user'] or not messages_by_role['assistant']:
                        print(f"Skipping conversation {idx} - missing user or assistant messages")
                        continue
                    
                    input_text = messages_by_role['user'][-1]['content']
                    expected_output = messages_by_role['assistant'][-1]['content']
                    prompt = f"{messages_by_role['system']}\n\n{input_text}" if messages_by_role['system'] else input_text
                    
                    response = br_runtime.invoke_model(
                        modelId=model_id,
                        body=json.dumps({
                            'prompt': prompt,
                            'max_tokens': 512,
                            'temperature': 0.5,
                            'top_p': 0.9,
                        }),
                        accept='application/json',
                        contentType='application/json'
                    )
                    
                    response_body = json.loads(response['body'].read())
                    model_output = response_body['generation']
                    
                    results.append({
                        'Question': input_text,
                        'Expected Answer': expected_output,
                        'Model Prediction': model_output,
                        'Input Tokens': response_body.get('prompt_token_count', 0),
                        'Output Tokens': response_body.get('generation_token_count', 0)
                    })
                    
                    print(f"Processed conversation {idx}/{len(conversations)}")
                    
                except Exception as e:
                    print(f"Error processing conversation {idx}: {str(e)}")
                    continue
        
        if not results:
            print("No results generated!")
            return pd.DataFrame()
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"Error running tests: {str(e)}")
        return pd.DataFrame()

def run_model_evaluation(model_id, s3_uri, num_samples=10, batch_size=5, display_examples=None):
    print("Starting model evaluation...")
    comparison_df = run_tests_from_s3(
        model_id=model_id,
        s3_uri=s3_uri,
        num_samples=num_samples,
        batch_size=batch_size
    )

    if not comparison_df.empty:
        display_results(comparison_df, num_examples=display_examples)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f'model_predictions_comparison_{timestamp}.csv'
        comparison_df.to_csv(csv_filename, index=False)
        print(f"\nResults saved to {csv_filename}")
        
        return comparison_df
    else:
        print("No data to analyze")
        return None

