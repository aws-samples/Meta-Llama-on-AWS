"""
Unit tests for SageMaker endpoint deployment script.

These tests verify the deployment, validation, and error handling
functionality without actually creating AWS resources.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError

# Import functions to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deployment.deploy_llama3_lmi import (
    deploy_model,
    wait_for_endpoint,
    check_endpoint_exists,
    get_or_create_sagemaker_role,
    get_hf_token,
    get_model_env
)


class TestGetHfToken:
    """Test suite for get_hf_token function."""
    
    @patch.dict('os.environ', {'HF_TOKEN': 'test_token_123'})
    def test_get_hf_token_from_hf_token_env(self):
        """Test getting token from HF_TOKEN environment variable."""
        token = get_hf_token()
        assert token == 'test_token_123'
    
    @patch.dict('os.environ', {'HUGGING_FACE_HUB_TOKEN': 'test_token_456'}, clear=True)
    def test_get_hf_token_from_hugging_face_hub_token_env(self):
        """Test getting token from HUGGING_FACE_HUB_TOKEN environment variable."""
        token = get_hf_token()
        assert token == 'test_token_456'
    
    @patch.dict('os.environ', {}, clear=True)
    def test_get_hf_token_missing_raises_error(self):
        """Test that missing token raises ValueError."""
        with pytest.raises(ValueError, match="HuggingFace token not found"):
            get_hf_token()


class TestGetModelEnv:
    """Test suite for get_model_env function."""
    
    @patch('deployment.deploy_llama3_lmi.get_hf_token')
    def test_get_model_env_returns_correct_structure(self, mock_get_token):
        """Test that get_model_env returns correct environment variables."""
        mock_get_token.return_value = 'test_token'
        
        env = get_model_env()
        
        assert env['HF_MODEL_ID'] == 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        assert env['HF_TOKEN'] == 'test_token'
        assert env['OPTION_ROLLING_BATCH'] == 'vllm'
        assert env['OPTION_ENABLE_AUTO_TOOL_CHOICE'] == 'true'
        assert env['OPTION_TOOL_CALL_PARSER'] == 'llama3_json'
        assert env['OPTION_MAX_MODEL_LEN'] == '8192'


class TestCheckEndpointExists:
    """Test suite for check_endpoint_exists function."""
    
    def test_check_endpoint_exists_returns_true_for_in_service(self):
        """Test that InService endpoint returns True."""
        mock_client = MagicMock()
        mock_client.describe_endpoint.return_value = {'EndpointStatus': 'InService'}
        
        result = check_endpoint_exists(mock_client, 'test-endpoint')
        assert result is True
    
    def test_check_endpoint_exists_returns_wait_for_creating(self):
        """Test that Creating endpoint returns 'wait'."""
        mock_client = MagicMock()
        mock_client.describe_endpoint.return_value = {'EndpointStatus': 'Creating'}
        
        result = check_endpoint_exists(mock_client, 'test-endpoint')
        assert result == 'wait'
    
    def test_check_endpoint_exists_returns_none_for_not_found(self):
        """Test that non-existent endpoint returns None."""
        mock_client = MagicMock()
        error_response = {'Error': {'Code': 'ValidationException', 'Message': 'Could not find endpoint'}}
        mock_client.describe_endpoint.side_effect = ClientError(error_response, 'DescribeEndpoint')
        
        result = check_endpoint_exists(mock_client, 'test-endpoint')
        assert result is None


class TestWaitForEndpoint:
    """Test suite for wait_for_endpoint function."""
    
    @patch('deployment.deploy_llama3_lmi.time.sleep')
    def test_wait_for_endpoint_success(self, mock_sleep):
        """Test successful wait for endpoint to reach InService."""
        mock_sagemaker = MagicMock()
        
        # Simulate endpoint transitioning to InService
        mock_sagemaker.describe_endpoint.side_effect = [
            {'EndpointStatus': 'Creating'},
            {'EndpointStatus': 'Creating'},
            {'EndpointStatus': 'InService'}
        ]
        
        # Execute - should return endpoint name
        result = wait_for_endpoint(mock_sagemaker, "test-endpoint")
        
        # Verify
        assert result == "test-endpoint"
        assert mock_sagemaker.describe_endpoint.call_count == 3
    
    @patch('deployment.deploy_llama3_lmi.time.sleep')
    def test_wait_for_endpoint_failed_status(self, mock_sleep):
        """Test handling of Failed endpoint status."""
        mock_sagemaker = MagicMock()
        
        # Endpoint fails
        mock_sagemaker.describe_endpoint.return_value = {
            'EndpointStatus': 'Failed',
            'FailureReason': 'Model download failed'
        }
        
        # Execute - should return None
        result = wait_for_endpoint(mock_sagemaker, "test-endpoint")
        
        # Verify
        assert result is None


class TestGetOrCreateSagemakerRole:
    """Test suite for get_or_create_sagemaker_role function."""
    
    @patch('deployment.deploy_llama3_lmi.boto3.client')
    def test_get_existing_role(self, mock_boto_client):
        """Test finding an existing SageMaker role."""
        mock_iam = MagicMock()
        mock_boto_client.return_value = mock_iam
        
        mock_iam.get_role.return_value = {
            'Role': {'Arn': 'arn:aws:iam::123456789:role/SageMakerExecutionRole'}
        }
        
        # Execute
        role_arn = get_or_create_sagemaker_role()
        
        # Verify
        assert role_arn == 'arn:aws:iam::123456789:role/SageMakerExecutionRole'
        mock_iam.get_role.assert_called()
    
    @patch('deployment.deploy_llama3_lmi.boto3.client')
    @patch('deployment.deploy_llama3_lmi.time.sleep')
    def test_create_new_role(self, mock_sleep, mock_boto_client):
        """Test creating a new SageMaker role when none exists."""
        from botocore.exceptions import ClientError
        
        mock_iam = MagicMock()
        mock_boto_client.return_value = mock_iam
        
        # Simulate no existing roles - use real ClientError
        error_response = {'Error': {'Code': 'NoSuchEntity', 'Message': 'Role not found'}}
        mock_iam.get_role.side_effect = ClientError(error_response, 'GetRole')
        
        # Mock successful role creation
        mock_iam.create_role.return_value = {
            'Role': {'Arn': 'arn:aws:iam::123456789:role/SageMakerExecutionRole'}
        }
        
        # Execute
        role_arn = get_or_create_sagemaker_role()
        
        # Verify
        assert role_arn == 'arn:aws:iam::123456789:role/SageMakerExecutionRole'
        mock_iam.create_role.assert_called_once()
        mock_iam.attach_role_policy.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
