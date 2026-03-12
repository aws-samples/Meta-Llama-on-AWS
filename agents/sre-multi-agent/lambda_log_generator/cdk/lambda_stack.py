#!/usr/bin/env python3
"""
CDK Stack for Bank Log Generator Lambda with API Gateway.

This stack creates:
- Lambda function to generate bank logs
- API Gateway REST API with Cognito authorization
- CloudWatch Log Group for bank logs
- IAM permissions
"""

from aws_cdk import (
    Stack,
    Duration,
    aws_lambda as lambda_,
    aws_apigateway as apigw,
    aws_iam as iam,
    aws_logs as logs,
    aws_cognito as cognito,
    CfnOutput,
)
from constructs import Construct


class BankLogGeneratorStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        user_pool_id: str,
        user_pool_arn: str,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # CloudWatch Log Group for bank logs (use existing)
        bank_log_group = logs.LogGroup.from_log_group_name(
            self,
            "BankLogGroup",
            log_group_name="/aws/banking/system-logs",
        )

        # Lambda execution role
        lambda_role = iam.Role(
            self,
            "LambdaExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )

        # Grant Lambda permission to write to CloudWatch Logs
        bank_log_group.grant_write(lambda_role)

        # Lambda function
        log_generator_lambda = lambda_.Function(
            self,
            "BankLogGenerator",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="lambda_function.lambda_handler",
            code=lambda_.Code.from_asset(
                "../",
                exclude=["cdk", "cdk.out", ".venv", "*.md", "*.sh", "test_*"]
            ),
            role=lambda_role,
            timeout=Duration.minutes(5),
            memory_size=512,
            environment={
                "LOG_GROUP_NAME": bank_log_group.log_group_name,
                "LOG_STREAM_PREFIX": "service",
            },
        )

        # API Gateway with Cognito authorizer
        api = apigw.RestApi(
            self,
            "BankLogGeneratorApi",
            rest_api_name="Bank Log Generator API",
            description="API to trigger bank log generation",
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
                allow_headers=[
                    "Content-Type",
                    "X-Amz-Date",
                    "Authorization",
                    "X-Api-Key",
                    "X-Amz-Security-Token",
                ],
            ),
        )

        # Cognito authorizer
        user_pool = cognito.UserPool.from_user_pool_arn(
            self,
            "UserPool",
            user_pool_arn
        )
        
        authorizer = apigw.CognitoUserPoolsAuthorizer(
            self,
            "CognitoAuthorizer",
            cognito_user_pools=[user_pool],
        )

        # Lambda integration
        lambda_integration = apigw.LambdaIntegration(
            log_generator_lambda,
            proxy=True,
        )

        # API resource and method
        generate_resource = api.root.add_resource("generate")
        generate_resource.add_method(
            "POST",
            lambda_integration,
            authorizer=authorizer,
            authorization_type=apigw.AuthorizationType.COGNITO,
        )

        # Outputs
        CfnOutput(
            self,
            "ApiUrl",
            value=api.url,
            description="Bank Log Generator API URL",
        )

        CfnOutput(
            self,
            "ApiEndpoint",
            value=f"{api.url}generate",
            description="Bank Log Generator API Endpoint",
        )

        CfnOutput(
            self,
            "LambdaFunctionName",
            value=log_generator_lambda.function_name,
            description="Lambda Function Name",
        )

        CfnOutput(
            self,
            "LogGroupName",
            value=bank_log_group.log_group_name,
            description="CloudWatch Log Group Name",
        )
