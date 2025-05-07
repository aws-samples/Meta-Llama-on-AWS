import boto3
import json
import os
import subprocess
import shutil
from botocore.exceptions import ClientError

def handle_client_error(func, *args, **kwargs):
    """
    Handle AWS client errors, specifically NoSuchEntity
    """
    try:
        return func(*args, **kwargs)
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            return None
        raise

def attach_policies(iam_client, role_name, policies_config):
    """
    Attach policies to the IAM role
    """
    for policy_name, policy_doc in policies_config.get('inline_policies', {}).items():
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName=policy_name,
            PolicyDocument=json.dumps(policy_doc)
        )
        print(f"Attached inline policy: {policy_name}")

    for policy_arn in policies_config.get('managed_policies', []):
        try:
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy_arn
            )
            print(f"Attached managed policy: {policy_arn}")
        except ClientError as e:
            if e.response['Error']['Code'] != 'EntityAlreadyExists':
                raise

def create_lambda_execution_role(role_name, trust_relationship, policies_config):
    """
    Create or update an IAM role with specified trust relationship and policies using the policies_config approach
    """
    iam = boto3.client('iam')
    
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_relationship),
            Description="Execution role for Lambda function and Bedrock model import jobs"
        )
        
        role_arn = response['Role']['Arn']
        print(f"Created IAM role: {role_arn}")
        
        attach_policies(iam, role_name, policies_config)
        
        return role_arn
    
    except ClientError as e:
        if e.response['Error']['Code'] == 'EntityAlreadyExists':
            print(f"IAM role {role_name} already exists. Retrieving its ARN.")
            role = iam.get_role(RoleName=role_name)
            
            iam.update_assume_role_policy(
                RoleName=role_name,
                PolicyDocument=json.dumps(trust_relationship)
            )
            
            attach_policies(iam, role_name, policies_config)
            
            return role['Role']['Arn']
        raise

def create_or_update_role(role_name, trust_relationship, permission_policy, iam_client=None, account_id=None):
    """
    Create or update an IAM role with a single permission policy
    """
    iam = iam_client or boto3.client('iam')
    account_id = account_id or boto3.client('sts').get_caller_identity()['Account']
    
    # Check and update/create role
    role = handle_client_error(iam.get_role, RoleName=role_name)
    if role:
        iam.update_assume_role_policy(
            RoleName=role_name,
            PolicyDocument=json.dumps(trust_relationship)
        )
        print(f"Updated existing role: {role_name}")
    else:
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_relationship)
        )
        print(f"Created new role: {role_name}")

    # Handle policy
    policy_name = f"{role_name}Policy"
    policy_arn = f"arn:aws:iam::{account_id}:policy/{policy_name}"
    
    # Attach or update policy
    policy = handle_client_error(iam.get_policy, PolicyArn=policy_arn)
    if policy:
        iam.create_policy_version(
            PolicyArn=policy_arn,
            PolicyDocument=json.dumps(permission_policy),
            SetAsDefault=True
        )
        # Cleanup old versions
        versions = iam.list_policy_versions(PolicyArn=policy_arn)['Versions']
        for version in versions:
            if not version['IsDefaultVersion']:
                iam.delete_policy_version(
                    PolicyArn=policy_arn,
                    VersionId=version['VersionId']
                )
        print(f"Updated existing policy: {policy_name}")
    else:
        iam.create_policy(
            PolicyName=policy_name,
            PolicyDocument=json.dumps(permission_policy)
        )
        print(f"Created new policy: {policy_name}")

    # Attach policy to role if not already attached
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn=policy_arn
    )
    print(f"Attached policy to role: {role_name}")

    return iam.get_role(RoleName=role_name)['Role']['Arn']



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
