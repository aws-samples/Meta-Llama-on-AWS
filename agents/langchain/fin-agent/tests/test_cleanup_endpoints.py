"""
Unit tests for cleanup_project.py SageMaker endpoint detection and cleanup.

Tests verify that cleanup_sagemaker_endpoints() correctly detects 8B-only,
70B-only, and both endpoints, handles non-existent endpoints gracefully,
and performs deletion when the user confirms.

Validates: Requirements 7.3
"""

import pytest
from unittest.mock import patch, MagicMock, call

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from botocore.exceptions import ClientError

from cleanup_project import cleanup_sagemaker_endpoints


def _make_validation_error():
    """Create a ClientError with ValidationException (endpoint not found)."""
    return ClientError(
        {"Error": {"Code": "ValidationException", "Message": "Could not find endpoint"}},
        "DescribeEndpoint",
    )


DESCRIBE_ENDPOINT_RESPONSE = {
    "EndpointStatus": "InService",
    "EndpointConfigName": "config-name-123",
}

DESCRIBE_ENDPOINT_CONFIG_RESPONSE = {
    "ProductionVariants": [
        {"InstanceType": "ml.g5.2xlarge", "ModelName": "model-name-123"}
    ]
}


class TestCleanupSagemakerEndpoints:
    """Tests for cleanup_sagemaker_endpoints() in cleanup_project.py."""

    @patch("cleanup_project.boto3")
    def test_no_endpoints_exist(self, mock_boto3):
        """Both describe_endpoint calls raise ValidationException — function completes without error."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.describe_endpoint.side_effect = [
            _make_validation_error(),
            _make_validation_error(),
        ]

        cleanup_sagemaker_endpoints()

        assert mock_client.describe_endpoint.call_count == 2
        mock_client.describe_endpoint.assert_any_call(EndpointName="llama3-lmi-agent")
        mock_client.describe_endpoint.assert_any_call(EndpointName="llama3-70b-lmi-agent")

    @patch("builtins.input", return_value="no")
    @patch("cleanup_project.boto3")
    def test_8b_only_exists(self, mock_boto3, mock_input):
        """Only 8B endpoint exists, 70B raises ValidationException. User declines deletion."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.describe_endpoint.side_effect = [
            DESCRIBE_ENDPOINT_RESPONSE,
            _make_validation_error(),
        ]
        mock_client.describe_endpoint_config.return_value = DESCRIBE_ENDPOINT_CONFIG_RESPONSE

        cleanup_sagemaker_endpoints()

        assert mock_client.describe_endpoint.call_count == 2
        mock_client.describe_endpoint.assert_any_call(EndpointName="llama3-lmi-agent")
        mock_client.describe_endpoint.assert_any_call(EndpointName="llama3-70b-lmi-agent")
        mock_client.delete_endpoint.assert_not_called()

    @patch("builtins.input", return_value="no")
    @patch("cleanup_project.boto3")
    def test_70b_only_exists(self, mock_boto3, mock_input):
        """Only 70B endpoint exists, 8B raises ValidationException. User declines deletion."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.describe_endpoint.side_effect = [
            _make_validation_error(),
            DESCRIBE_ENDPOINT_RESPONSE,
        ]
        mock_client.describe_endpoint_config.return_value = DESCRIBE_ENDPOINT_CONFIG_RESPONSE

        cleanup_sagemaker_endpoints()

        assert mock_client.describe_endpoint.call_count == 2
        mock_client.describe_endpoint.assert_any_call(EndpointName="llama3-lmi-agent")
        mock_client.describe_endpoint.assert_any_call(EndpointName="llama3-70b-lmi-agent")
        mock_client.delete_endpoint.assert_not_called()

    @patch("builtins.input", return_value="no")
    @patch("cleanup_project.boto3")
    def test_both_endpoints_exist(self, mock_boto3, mock_input):
        """Both endpoints exist. User declines deletion for both."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.describe_endpoint.side_effect = [
            DESCRIBE_ENDPOINT_RESPONSE,
            DESCRIBE_ENDPOINT_RESPONSE,
        ]
        mock_client.describe_endpoint_config.return_value = DESCRIBE_ENDPOINT_CONFIG_RESPONSE

        cleanup_sagemaker_endpoints()

        assert mock_client.describe_endpoint.call_count == 2
        mock_client.describe_endpoint.assert_any_call(EndpointName="llama3-lmi-agent")
        mock_client.describe_endpoint.assert_any_call(EndpointName="llama3-70b-lmi-agent")
        mock_client.delete_endpoint.assert_not_called()

    @patch("builtins.input", return_value="yes")
    @patch("cleanup_project.boto3")
    def test_deletion_confirmed(self, mock_boto3, mock_input):
        """One endpoint exists, user confirms 'yes'. Verify delete calls are made."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.describe_endpoint.side_effect = [
            DESCRIBE_ENDPOINT_RESPONSE,
            _make_validation_error(),
        ]
        mock_client.describe_endpoint_config.return_value = DESCRIBE_ENDPOINT_CONFIG_RESPONSE

        cleanup_sagemaker_endpoints()

        mock_client.delete_endpoint.assert_called_once_with(EndpointName="llama3-lmi-agent")
        mock_client.delete_endpoint_config.assert_called_once_with(EndpointConfigName="config-name-123")
        mock_client.delete_model.assert_called_once_with(ModelName="model-name-123")
