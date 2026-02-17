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

from deployment.deploy_endpoint import (
    deploy_endpoint,
    wait_for_endpoint,
    validate_endpoint,
    delete_endpoint
)


class TestDeployEndpoint:
    """Test suite for deploy_endpoint function."""
    
    def test_deploy_endpoint_requires_role_arn(self):
        """Test that deploy_endpoint raises ValueError without role_arn."""
        with pytest.raises(ValueError, match="role_arn is required"):
            deploy_endpoint(endpoint_name="test-endpoint", role_arn=None)
    
    @patch('deployment.deploy_endpoint.boto3.client')
    @patch('deployment.deploy_endpoint.wait_for_endpoint')
    @patch('deployment.deploy_endpoint.validate_endpoint')
    def test_deploy_endpoint_success(self, mock_validate, mock_wait, mock_boto_client):
        """Test successful endpoint deployment."""
        # Setup mocks
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_sagemaker.meta.region_name = 'us-west-2'
        
        mock_sagemaker.create_model.return_value = {
            'ModelArn': 'arn:aws:sagemaker:us-west-2:123456789:model/test-model'
        }
        mock_sagemaker.create_endpoint_config.return_value = {
            'EndpointConfigArn': 'arn:aws:sagemaker:us-west-2:123456789:endpoint-config/test-config'
        }
        mock_sagemaker.create_endpoint.return_value = {
            'EndpointArn': 'arn:aws:sagemaker:us-west-2:123456789:endpoint/test-endpoint'
        }
        
        mock_validate.return_value = {"valid": True, "message": "Success"}
        
        # Execute
        result = deploy_endpoint(
            endpoint_name="test-endpoint",
            role_arn="arn:aws:iam::123456789:role/TestRole"
        )
        
        # Verify
        assert result['status'] == 'success'
        assert result['endpoint_name'] == 'test-endpoint'
        assert result['region'] == 'us-west-2'
        assert 'model_name' in result
        assert 'endpoint_config_name' in result
        
        # Verify AWS API calls
        mock_sagemaker.create_model.assert_called_once()
        mock_sagemaker.create_endpoint_config.assert_called_once()
        mock_sagemaker.create_endpoint.assert_called_once()
        mock_wait.assert_called_once()
        mock_validate.assert_called_once()
    
    @patch('deployment.deploy_endpoint.boto3.client')
    def test_deploy_endpoint_resource_limit_exceeded(self, mock_boto_client):
        """Test handling of ResourceLimitExceeded error."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_sagemaker.meta.region_name = 'us-west-2'
        
        # Simulate ResourceLimitExceeded error
        error_response = {
            'Error': {
                'Code': 'ResourceLimitExceeded',
                'Message': 'Instance quota exceeded'
            }
        }
        mock_sagemaker.create_model.side_effect = ClientError(error_response, 'CreateModel')
        
        # Execute
        result = deploy_endpoint(
            endpoint_name="test-endpoint",
            role_arn="arn:aws:iam::123456789:role/TestRole"
        )
        
        # Verify
        assert result['status'] == 'error'
        assert 'quota exceeded' in result['message'].lower()
    
    @patch('deployment.deploy_endpoint.boto3.client')
    def test_deploy_endpoint_validation_exception(self, mock_boto_client):
        """Test handling of ValidationException error."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_sagemaker.meta.region_name = 'us-west-2'
        
        # Simulate ValidationException error
        error_response = {
            'Error': {
                'Code': 'ValidationException',
                'Message': 'Invalid instance type'
            }
        }
        mock_sagemaker.create_model.side_effect = ClientError(error_response, 'CreateModel')
        
        # Execute
        result = deploy_endpoint(
            endpoint_name="test-endpoint",
            role_arn="arn:aws:iam::123456789:role/TestRole"
        )
        
        # Verify
        assert result['status'] == 'error'
        assert 'Invalid configuration' in result['message']
    
    @patch('deployment.deploy_endpoint.boto3.client')
    def test_deploy_endpoint_access_denied(self, mock_boto_client):
        """Test handling of AccessDeniedException error."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_sagemaker.meta.region_name = 'us-west-2'
        
        # Simulate AccessDeniedException error
        error_response = {
            'Error': {
                'Code': 'AccessDeniedException',
                'Message': 'User not authorized'
            }
        }
        mock_sagemaker.create_model.side_effect = ClientError(error_response, 'CreateModel')
        
        # Execute
        result = deploy_endpoint(
            endpoint_name="test-endpoint",
            role_arn="arn:aws:iam::123456789:role/TestRole"
        )
        
        # Verify
        assert result['status'] == 'error'
        assert 'IAM permission denied' in result['message']
    
    @patch('deployment.deploy_endpoint.boto3.client')
    @patch('deployment.deploy_endpoint.wait_for_endpoint')
    @patch('deployment.deploy_endpoint.validate_endpoint')
    def test_deploy_endpoint_validation_failure(self, mock_validate, mock_wait, mock_boto_client):
        """Test handling when endpoint validation fails."""
        # Setup mocks
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_sagemaker.meta.region_name = 'us-west-2'
        
        mock_sagemaker.create_model.return_value = {'ModelArn': 'arn:test'}
        mock_sagemaker.create_endpoint_config.return_value = {'EndpointConfigArn': 'arn:test'}
        mock_sagemaker.create_endpoint.return_value = {'EndpointArn': 'arn:test'}
        
        mock_validate.return_value = {
            "valid": False,
            "message": "Endpoint not responding"
        }
        
        # Execute
        result = deploy_endpoint(
            endpoint_name="test-endpoint",
            role_arn="arn:aws:iam::123456789:role/TestRole"
        )
        
        # Verify
        assert result['status'] == 'error'
        assert 'validation failed' in result['message'].lower()


class TestWaitForEndpoint:
    """Test suite for wait_for_endpoint function."""
    
    @patch('deployment.deploy_endpoint.boto3.client')
    @patch('deployment.deploy_endpoint.time.sleep')
    def test_wait_for_endpoint_success(self, mock_sleep, mock_boto_client):
        """Test successful wait for endpoint to reach InService."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        
        # Simulate endpoint transitioning to InService
        mock_sagemaker.describe_endpoint.side_effect = [
            {'EndpointStatus': 'Creating'},
            {'EndpointStatus': 'Creating'},
            {'EndpointStatus': 'InService'}
        ]
        
        # Execute - should not raise
        wait_for_endpoint("test-endpoint", timeout=300)
        
        # Verify
        assert mock_sagemaker.describe_endpoint.call_count == 3
    
    @patch('deployment.deploy_endpoint.boto3.client')
    @patch('deployment.deploy_endpoint.time.sleep')
    @patch('deployment.deploy_endpoint.time.time')
    def test_wait_for_endpoint_timeout(self, mock_time, mock_sleep, mock_boto_client):
        """Test timeout when endpoint takes too long."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        
        # Simulate time passing
        mock_time.side_effect = [0, 100, 200, 301]  # Exceeds 300s timeout
        
        # Endpoint stays in Creating state
        mock_sagemaker.describe_endpoint.return_value = {'EndpointStatus': 'Creating'}
        
        # Execute - should raise TimeoutError
        with pytest.raises(TimeoutError, match="did not reach InService"):
            wait_for_endpoint("test-endpoint", timeout=300)
    
    @patch('deployment.deploy_endpoint.boto3.client')
    def test_wait_for_endpoint_failed_status(self, mock_boto_client):
        """Test handling of Failed endpoint status."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        
        # Endpoint fails
        mock_sagemaker.describe_endpoint.return_value = {
            'EndpointStatus': 'Failed',
            'FailureReason': 'Model download failed'
        }
        
        # Execute - should raise RuntimeError
        with pytest.raises(RuntimeError, match="failed to create"):
            wait_for_endpoint("test-endpoint")


class TestValidateEndpoint:
    """Test suite for validate_endpoint function."""
    
    @patch('deployment.deploy_endpoint.boto3.client')
    def test_validate_endpoint_success(self, mock_boto_client):
        """Test successful endpoint validation."""
        mock_runtime = MagicMock()
        mock_boto_client.return_value = mock_runtime
        
        # Mock successful response in OpenAI format
        mock_response = MagicMock()
        mock_response['Body'].read.return_value = json.dumps({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, I am working!"
                },
                "finish_reason": "stop"
            }]
        }).encode('utf-8')
        mock_runtime.invoke_endpoint.return_value = mock_response
        
        # Execute
        result = validate_endpoint("test-endpoint")
        
        # Verify
        assert result['valid'] is True
        assert 'functional' in result['message'].lower()
        assert 'response' in result
    
    @patch('deployment.deploy_endpoint.boto3.client')
    def test_validate_endpoint_invalid_response_structure(self, mock_boto_client):
        """Test handling of invalid response structure."""
        mock_runtime = MagicMock()
        mock_boto_client.return_value = mock_runtime
        
        # Mock invalid response (missing choices)
        mock_response = MagicMock()
        mock_response['Body'].read.return_value = json.dumps(
            {'error': 'Invalid'}
        ).encode('utf-8')
        mock_runtime.invoke_endpoint.return_value = mock_response
        
        # Execute
        result = validate_endpoint("test-endpoint")
        
        # Verify
        assert result['valid'] is False
        assert 'choices' in result['message'].lower()
    
    @patch('deployment.deploy_endpoint.boto3.client')
    def test_validate_endpoint_missing_generated_text(self, mock_boto_client):
        """Test handling of response missing message field."""
        mock_runtime = MagicMock()
        mock_boto_client.return_value = mock_runtime
        
        # Mock response without message field
        mock_response = MagicMock()
        mock_response['Body'].read.return_value = json.dumps({
            "choices": [{
                "some_other_field": "value"
            }]
        }).encode('utf-8')
        mock_runtime.invoke_endpoint.return_value = mock_response
        
        # Execute
        result = validate_endpoint("test-endpoint")
        
        # Verify
        assert result['valid'] is False
        assert 'message' in result['message'].lower()
    
    @patch('deployment.deploy_endpoint.boto3.client')
    def test_validate_endpoint_client_error(self, mock_boto_client):
        """Test handling of AWS client error during validation."""
        mock_runtime = MagicMock()
        mock_boto_client.return_value = mock_runtime
        
        # Simulate client error
        error_response = {
            'Error': {
                'Code': 'ModelError',
                'Message': 'Model failed to load'
            }
        }
        mock_runtime.invoke_endpoint.side_effect = ClientError(error_response, 'InvokeEndpoint')
        
        # Execute
        result = validate_endpoint("test-endpoint")
        
        # Verify
        assert result['valid'] is False
        assert 'invocation failed' in result['message'].lower()


class TestDeleteEndpoint:
    """Test suite for delete_endpoint function."""
    
    @patch('deployment.deploy_endpoint.boto3.client')
    def test_delete_endpoint_success(self, mock_boto_client):
        """Test successful endpoint deletion."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        
        # Mock endpoint info
        mock_sagemaker.describe_endpoint.return_value = {
            'EndpointConfigName': 'test-config'
        }
        mock_sagemaker.describe_endpoint_config.return_value = {
            'ProductionVariants': [{'ModelName': 'test-model'}]
        }
        
        # Execute
        result = delete_endpoint("test-endpoint")
        
        # Verify
        assert result['status'] == 'success'
        mock_sagemaker.delete_endpoint.assert_called_once()
        mock_sagemaker.delete_endpoint_config.assert_called_once()
        mock_sagemaker.delete_model.assert_called_once()
    
    @patch('deployment.deploy_endpoint.boto3.client')
    def test_delete_endpoint_client_error(self, mock_boto_client):
        """Test handling of error during deletion."""
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        
        # Simulate error
        error_response = {
            'Error': {
                'Code': 'ResourceNotFound',
                'Message': 'Endpoint not found'
            }
        }
        mock_sagemaker.describe_endpoint.side_effect = ClientError(error_response, 'DescribeEndpoint')
        
        # Execute
        result = delete_endpoint("test-endpoint")
        
        # Verify
        assert result['status'] == 'error'
        assert 'failed' in result['message'].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
