"""
Unit tests for the 70B SageMaker endpoint deployment script.

Tests verify get_model_env() returns correct environment variables
for both HuggingFace and S3 model sources, including S3 URI validation.
"""

import pytest
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deployment.deploy_llama3_70b import get_model_env


class TestGetModelEnv:
    """Test suite for get_model_env in the 70B deployment script."""

    @patch('deployment.deploy_llama3_70b.get_hf_token')
    def test_huggingface_source_returns_all_env_vars(self, mock_get_token):
        """When no S3 URI is provided, env should use HF model ID and include HF_TOKEN."""
        mock_get_token.return_value = 'hf_test_token'

        env = get_model_env()

        assert env['HF_MODEL_ID'] == 'meta-llama/Meta-Llama-3.1-70B-Instruct'
        assert env['HF_TOKEN'] == 'hf_test_token'
        assert env['OPTION_ROLLING_BATCH'] == 'vllm'
        assert env['OPTION_ENABLE_AUTO_TOOL_CHOICE'] == 'true'
        assert env['OPTION_TOOL_CALL_PARSER'] == 'llama3_json'
        assert env['OPTION_MAX_ROLLING_BATCH_SIZE'] == '4'
        assert env['OPTION_MAX_MODEL_LEN'] == '32768'
        assert env['OPTION_DTYPE'] == 'fp16'
        assert env['TENSOR_PARALLEL_DEGREE'] == '8'

    def test_s3_source_sets_model_id_to_s3_uri(self):
        """When S3 URI is provided, HF_MODEL_ID should be the S3 URI."""
        env = get_model_env(model_s3_uri='s3://my-bucket/llama-70b/')

        assert env['HF_MODEL_ID'] == 's3://my-bucket/llama-70b/'

    def test_s3_source_omits_hf_token(self):
        """When S3 URI is provided, HF_TOKEN should not be in the env dict."""
        env = get_model_env(model_s3_uri='s3://my-bucket/llama-70b/')

        assert 'HF_TOKEN' not in env

    def test_s3_source_includes_inference_params(self):
        """S3 source should still include all inference configuration."""
        env = get_model_env(model_s3_uri='s3://my-bucket/llama-70b/')

        assert env['OPTION_ROLLING_BATCH'] == 'vllm'
        assert env['OPTION_ENABLE_AUTO_TOOL_CHOICE'] == 'true'
        assert env['OPTION_TOOL_CALL_PARSER'] == 'llama3_json'
        assert env['OPTION_MAX_ROLLING_BATCH_SIZE'] == '4'
        assert env['OPTION_MAX_MODEL_LEN'] == '32768'
        assert env['OPTION_DTYPE'] == 'fp16'
        assert env['TENSOR_PARALLEL_DEGREE'] == '8'

    def test_invalid_s3_uri_missing_prefix_raises_error(self):
        """S3 URI without 's3://' prefix should raise ValueError."""
        with pytest.raises(ValueError, match="must start with 's3://'"):
            get_model_env(model_s3_uri='https://my-bucket/llama-70b/')

    def test_invalid_s3_uri_no_bucket_raises_error(self):
        """S3 URI with no bucket name after 's3://' should raise ValueError."""
        with pytest.raises(ValueError, match="must include a bucket name"):
            get_model_env(model_s3_uri='s3://')

    def test_invalid_s3_uri_slash_after_prefix_raises_error(self):
        """S3 URI like 's3:///path' (no bucket) should raise ValueError."""
        with pytest.raises(ValueError, match="must include a bucket name"):
            get_model_env(model_s3_uri='s3:///no-bucket-here')

    def test_s3_uri_bucket_only_is_valid(self):
        """S3 URI with just a bucket name (no trailing path) should be valid."""
        env = get_model_env(model_s3_uri='s3://my-bucket')

        assert env['HF_MODEL_ID'] == 's3://my-bucket'

    @patch('deployment.deploy_llama3_70b.get_hf_token')
    def test_none_s3_uri_uses_huggingface(self, mock_get_token):
        """Explicitly passing None should behave like HuggingFace source."""
        mock_get_token.return_value = 'token'

        env = get_model_env(model_s3_uri=None)

        assert env['HF_MODEL_ID'] == 'meta-llama/Meta-Llama-3.1-70B-Instruct'
        assert 'HF_TOKEN' in env


from deployment.deploy_llama3_70b import check_instance_quota, main


class TestCheckInstanceQuota:
    """Test suite for check_instance_quota in the 70B deployment script."""

    def _make_paginator(self, quotas):
        """Helper: build a mock paginator that yields a single page of quotas."""
        class FakePaginator:
            def paginate(self, **kwargs):
                return [{"Quotas": quotas}]
        return FakePaginator()

    @patch("deployment.deploy_llama3_70b.boto3")
    def test_sufficient_quota_returns_true(self, mock_boto3):
        """When the quota value is >= 1, check_instance_quota should return True."""
        paginator = self._make_paginator([
            {"QuotaName": "ml.g5.48xlarge for endpoint usage", "Value": 2}
        ])
        mock_client = mock_boto3.client.return_value
        mock_client.get_paginator.return_value = paginator

        result = check_instance_quota("ml.g5.48xlarge")

        assert result is True

    @patch("deployment.deploy_llama3_70b.boto3")
    def test_insufficient_quota_returns_false(self, mock_boto3):
        """When the quota value is 0, check_instance_quota should return False."""
        paginator = self._make_paginator([
            {"QuotaName": "ml.g5.48xlarge for endpoint usage", "Value": 0}
        ])
        mock_client = mock_boto3.client.return_value
        mock_client.get_paginator.return_value = paginator

        result = check_instance_quota("ml.g5.48xlarge")

        assert result is False

    @patch("deployment.deploy_llama3_70b.boto3")
    def test_quota_not_found_returns_true(self, mock_boto3):
        """When no matching quota is found in either applied or default quotas, return True (non-blocking)."""
        empty_paginator = self._make_paginator([])
        mock_client = mock_boto3.client.return_value
        mock_client.get_paginator.return_value = empty_paginator

        result = check_instance_quota("ml.g5.48xlarge")

        assert result is True

    @patch("deployment.deploy_llama3_70b.boto3")
    def test_api_exception_returns_true(self, mock_boto3):
        """When the service-quotas API raises an exception, return True (graceful fallback)."""
        mock_boto3.client.side_effect = Exception("Service unavailable")

        result = check_instance_quota("ml.g5.48xlarge")

        assert result is True


class TestMainArgParsing:
    """Test suite for main() argument parsing in the 70B deployment script."""

    @patch("deployment.deploy_llama3_70b.deploy_model")
    @patch("deployment.deploy_llama3_70b.check_instance_quota", return_value=True)
    @patch("builtins.input", return_value="yes")
    @patch("sys.argv", ["deploy_llama3_70b.py", "--model-s3-uri", "s3://my-bucket/llama-70b/"])
    def test_model_s3_uri_parsed_correctly(self, mock_input, mock_quota, mock_deploy):
        """When --model-s3-uri is provided, deploy_model should receive the S3 URI."""
        mock_deploy.return_value = "llama3-70b-lmi-agent"

        main()

        mock_deploy.assert_called_once_with("s3://my-bucket/llama-70b/")

    @patch("deployment.deploy_llama3_70b.deploy_model")
    @patch("deployment.deploy_llama3_70b.check_instance_quota", return_value=True)
    @patch("builtins.input", return_value="yes")
    @patch("sys.argv", ["deploy_llama3_70b.py"])
    def test_no_model_s3_uri_defaults_to_none(self, mock_input, mock_quota, mock_deploy):
        """When --model-s3-uri is not provided, deploy_model should receive None."""
        mock_deploy.return_value = "llama3-70b-lmi-agent"

        main()

        mock_deploy.assert_called_once_with(None)
