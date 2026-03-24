"""
Unit tests for the S3 model pre-download utility.

Tests cover CLI argument parsing, S3 access validation, and S3 URI output format.

Validates: Requirements 8.2, 8.3, 8.7
"""

import os
import pytest
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from botocore.exceptions import ClientError

from deployment.download_model_to_s3 import (
    parse_args,
    validate_s3_access,
    check_existing_model,
    download_and_upload,
    _resolve_sagemaker_default_bucket,
    _derive_s3_prefix,
    DEFAULT_MODEL_ID,
    DEFAULT_S3_PREFIX,
)


class TestParseArgs:
    """Test suite for parse_args() CLI argument parsing."""

    def test_s3_bucket_falls_back_to_sagemaker_default(self):
        """When --s3-bucket is omitted, parse_args should resolve the SageMaker default bucket."""
        with patch("deployment.download_model_to_s3._resolve_sagemaker_default_bucket",
                    return_value="sagemaker-us-west-2-123456789012"):
            args = parse_args([])
        assert args.s3_bucket == "sagemaker-us-west-2-123456789012"

    def test_explicit_s3_bucket_skips_default_resolution(self):
        """When --s3-bucket is provided, the SageMaker default bucket should not be resolved."""
        with patch("deployment.download_model_to_s3._resolve_sagemaker_default_bucket") as mock_resolve:
            args = parse_args(["--s3-bucket", "my-bucket"])
        mock_resolve.assert_not_called()
        assert args.s3_bucket == "my-bucket"

    def test_model_id_defaults_to_70b_instruct(self):
        """--model-id should default to Meta-Llama-3.1-70B-Instruct."""
        args = parse_args(["--s3-bucket", "my-bucket"])
        assert args.model_id == DEFAULT_MODEL_ID

    def test_s3_prefix_auto_derived_from_model_id(self):
        """When --s3-prefix is omitted, it should be auto-derived from --model-id."""
        args = parse_args(["--s3-bucket", "my-bucket"])
        assert args.s3_prefix == "Meta-Llama-3.1-70B-Instruct/"

    def test_s3_prefix_auto_derived_for_quantized_model(self):
        """Quantized model ID should produce a different S3 prefix than FP16."""
        args = parse_args([
            "--s3-bucket", "my-bucket",
            "--model-id", "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        ])
        assert args.s3_prefix == "Meta-Llama-3.1-70B-Instruct-AWQ-INT4/"

    def test_explicit_s3_prefix_overrides_auto_derivation(self):
        """When --s3-prefix is provided, it should override auto-derivation."""
        args = parse_args([
            "--s3-bucket", "my-bucket",
            "--s3-prefix", "custom-prefix/",
        ])
        assert args.s3_prefix == "custom-prefix/"

    def test_hf_token_falls_back_to_env_var(self):
        """When --hf-token is not provided, parse_args should read HF_TOKEN env var."""
        with patch.dict(os.environ, {"HF_TOKEN": "env_token_123"}, clear=False):
            args = parse_args(["--s3-bucket", "my-bucket"])
        assert args.hf_token == "env_token_123"

    def test_hf_token_none_when_no_env_and_no_arg(self):
        """When neither --hf-token nor HF_TOKEN env var is set, hf_token should be None."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure both env vars are absent
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
            args = parse_args(["--s3-bucket", "my-bucket"])
        assert args.hf_token is None

    def test_all_arguments_parsed_correctly(self):
        """All CLI arguments should be parsed when explicitly provided."""
        args = parse_args([
            "--model-id", "meta-llama/Meta-Llama-3.3-70B-Instruct",
            "--s3-bucket", "test-bucket",
            "--s3-prefix", "models/llama/",
            "--hf-token", "hf_my_token",
        ])
        assert args.model_id == "meta-llama/Meta-Llama-3.3-70B-Instruct"
        assert args.s3_bucket == "test-bucket"
        assert args.s3_prefix == "models/llama/"
        assert args.hf_token == "hf_my_token"


class TestValidateS3Access:
    """Test suite for validate_s3_access() with mocked boto3 S3 client."""

    def _make_client_error(self, code, message="error"):
        """Helper to create a botocore ClientError."""
        return ClientError(
            {"Error": {"Code": code, "Message": message}},
            "HeadBucket",
        )

    def test_accessible_bucket_returns_true(self):
        """validate_s3_access should return True for an accessible, writable bucket."""
        mock_s3 = MagicMock()
        # head_bucket succeeds, put_object succeeds, delete_object succeeds
        result = validate_s3_access(mock_s3, "my-bucket", "prefix/")
        assert result is True
        mock_s3.head_bucket.assert_called_once_with(Bucket="my-bucket")

    def test_nonexistent_bucket_returns_false(self):
        """validate_s3_access should return False when bucket does not exist (404)."""
        mock_s3 = MagicMock()
        mock_s3.head_bucket.side_effect = self._make_client_error("404")

        result = validate_s3_access(mock_s3, "no-such-bucket", "prefix/")
        assert result is False

    def test_access_denied_returns_false(self):
        """validate_s3_access should return False when access is denied (403)."""
        mock_s3 = MagicMock()
        mock_s3.head_bucket.side_effect = self._make_client_error("403")

        result = validate_s3_access(mock_s3, "private-bucket", "prefix/")
        assert result is False

    def test_write_test_failure_returns_false(self):
        """validate_s3_access should return False when put_object fails."""
        mock_s3 = MagicMock()
        # head_bucket succeeds, but put_object fails
        mock_s3.put_object.side_effect = self._make_client_error("AccessDenied", "Write denied")

        result = validate_s3_access(mock_s3, "read-only-bucket", "prefix/")
        assert result is False


class TestS3UriFormat:
    """Test suite for S3 URI output format from download_and_upload."""

    @patch("deployment.download_model_to_s3.boto3")
    @patch("deployment.download_model_to_s3.snapshot_download", create=True)
    def test_returns_correct_s3_uri_format(self, mock_snapshot, mock_boto3):
        """download_and_upload should return 's3://{bucket}/{prefix}'."""
        # We need to patch the import inside the function
        with patch("deployment.download_model_to_s3.tempfile.TemporaryDirectory") as mock_tmpdir:
            import tempfile as real_tempfile
            # Create a real temp dir for the test
            real_tmp = real_tempfile.mkdtemp()
            mock_tmpdir.return_value.__enter__ = MagicMock(return_value=real_tmp)
            mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)

            # snapshot_download returns the local dir path
            mock_snapshot.return_value = real_tmp

            # Mock the S3 client
            mock_s3_client = MagicMock()
            mock_boto3.client.return_value = mock_s3_client

            # Patch huggingface_hub.snapshot_download inside the function
            with patch("huggingface_hub.snapshot_download", return_value=real_tmp):
                result = download_and_upload(
                    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
                    s3_bucket="my-bucket",
                    s3_prefix="llama-70b-instruct/",
                    hf_token="hf_test",
                )

            assert result == "s3://my-bucket/llama-70b-instruct/"

            # Clean up
            import shutil
            shutil.rmtree(real_tmp, ignore_errors=True)


class TestResolveSagemakerDefaultBucket:
    """Test suite for _resolve_sagemaker_default_bucket()."""

    @patch("deployment.download_model_to_s3.sagemaker", create=True)
    def test_returns_default_bucket_name(self, mock_sagemaker_module):
        """Should return the bucket name from sagemaker.Session().default_bucket()."""
        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "sagemaker-us-west-2-123456789012"
        mock_sagemaker_module.Session.return_value = mock_session

        # We need to patch the import inside the function
        with patch.dict("sys.modules", {"sagemaker": mock_sagemaker_module}):
            result = _resolve_sagemaker_default_bucket()

        assert result == "sagemaker-us-west-2-123456789012"

    def test_exits_when_sagemaker_not_installed(self):
        """Should sys.exit(1) when the sagemaker package is not importable."""
        with patch.dict("sys.modules", {"sagemaker": None}):
            with pytest.raises(SystemExit):
                _resolve_sagemaker_default_bucket()


class TestCheckExistingModel:
    """Test suite for check_existing_model() with mocked S3 paginator."""

    def test_returns_true_when_files_exist(self):
        """Should return (True, count, size) when objects exist under the prefix."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "llama-70b-instruct/config.json", "Size": 1024},
                    {"Key": "llama-70b-instruct/model-00001.safetensors", "Size": 5_000_000_000},
                ]
            }
        ]

        exists, count, size = check_existing_model(mock_s3, "my-bucket", "llama-70b-instruct/")
        assert exists is True
        assert count == 2
        assert size == 5_000_001_024

    def test_returns_false_when_prefix_is_empty(self):
        """Should return (False, 0, 0) when no objects exist under the prefix."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{"Contents": []}]

        exists, count, size = check_existing_model(mock_s3, "my-bucket", "llama-70b-instruct/")
        assert exists is False
        assert count == 0
        assert size == 0

    def test_skips_hidden_files(self):
        """Should ignore hidden files (starting with '.') when counting."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "llama-70b-instruct/.gitattributes", "Size": 100},
                    {"Key": "llama-70b-instruct/config.json", "Size": 1024},
                ]
            }
        ]

        exists, count, size = check_existing_model(mock_s3, "my-bucket", "llama-70b-instruct/")
        assert exists is True
        assert count == 1
        assert size == 1024

    def test_returns_false_on_client_error(self):
        """Should return (False, 0, 0) when list_objects_v2 raises a ClientError."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}},
            "ListObjectsV2",
        )

        exists, count, size = check_existing_model(mock_s3, "my-bucket", "llama-70b-instruct/")
        assert exists is False
        assert count == 0


class TestForceFlag:
    """Test suite for the --force CLI flag."""

    def test_force_flag_defaults_to_false(self):
        """--force should default to False."""
        with patch("deployment.download_model_to_s3._resolve_sagemaker_default_bucket",
                    return_value="default-bucket"):
            args = parse_args([])
        assert args.force is False

    def test_force_flag_set_to_true(self):
        """--force should be True when provided."""
        args = parse_args(["--s3-bucket", "my-bucket", "--force"])
        assert args.force is True


class TestDeriveS3Prefix:
    """Test suite for _derive_s3_prefix() auto-derivation from model IDs."""

    def test_standard_model_id(self):
        """Standard org/model format should extract the model name."""
        assert _derive_s3_prefix("meta-llama/Meta-Llama-3.1-70B-Instruct") == "Meta-Llama-3.1-70B-Instruct/"

    def test_quantized_model_id(self):
        """Quantized model should produce a different prefix than FP16."""
        assert _derive_s3_prefix("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4") == "Meta-Llama-3.1-70B-Instruct-AWQ-INT4/"

    def test_no_org_prefix(self):
        """Model ID without org/ should use the whole string."""
        assert _derive_s3_prefix("my-custom-model") == "my-custom-model/"

    def test_llama_33_variant(self):
        """Llama 3.3 variant should get its own prefix."""
        assert _derive_s3_prefix("meta-llama/Llama-3.3-70B-Instruct") == "Llama-3.3-70B-Instruct/"

    def test_fp16_and_quantized_dont_collide(self):
        """FP16 and quantized variants of the same model must produce different prefixes."""
        fp16_prefix = _derive_s3_prefix("meta-llama/Meta-Llama-3.1-70B-Instruct")
        awq_prefix = _derive_s3_prefix("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
        assert fp16_prefix != awq_prefix
