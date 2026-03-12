#!/usr/bin/env python3
"""
CDK App for Bank Log Generator Lambda.

Usage:
    cdk deploy --context user_pool_id=us-west-2_XXXXX --context user_pool_arn=arn:aws:cognito-idp:...
"""

import os
from aws_cdk import App, Environment
from lambda_stack import BankLogGeneratorStack

app = App()

# Get user pool info from context (passed by deploy.sh) or environment
user_pool_id = app.node.try_get_context("user_pool_id") or os.environ.get("USER_POOL_ID")
user_pool_arn = app.node.try_get_context("user_pool_arn") or os.environ.get("USER_POOL_ARN")

if not user_pool_id or not user_pool_arn:
    raise ValueError(
        "user_pool_id and user_pool_arn are required. "
        "Pass via --context or set USER_POOL_ID and USER_POOL_ARN env vars."
    )

BankLogGeneratorStack(
    app,
    "BankLogGeneratorStack",
    user_pool_id=user_pool_id,
    user_pool_arn=user_pool_arn,
    env=Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-west-2"),
    ),
)

app.synth()
