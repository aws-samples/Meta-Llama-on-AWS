# RAG Recipes

This repository provides a *Cloudformation template* to deploy a SageMaker Environment where you can explore, test, and experiment Text-to-SQL using:
1. Llama 3
2. Llama 3.1 (us-west-2 only)

### Prerequisites
To interact with the models, you need to [request access to the models in the region you will use](https://console.aws.amazon.com/bedrock/home?#/modelaccess)*. Make sure to read and accept the end-user license agreements or EULA.

***Note:** To make sure that you have enough quotas to support your usage requirements, it's a best practice to monitor and manage your service quotas. Requests for Amazon SageMaker service quota increases are subject to review by AWS engineering teams. Also, service quota increase requests aren't immediately processed when you submit a request. After your request is processed, you receive an email notification.*

### Instance type quota increase

Complete the following steps:

- Open the [Service Quotas console](https://console.aws.amazon.com/servicequotas/).
- Choose Amazon SageMaker.
- Choose the service quota.
- Choose Request quota increase.

## Deployment

The solution is deployed using an AWS CloudFormation template with Amazon SageMaker Notebook Instance. To deploy the solution, use one of the following CloudFormation templates and follow the instructions below.

Per [guidance for workload isolation on AWS](https://aws.amazon.com/solutions/guidance/workload-isolation-on-aws/), it is recommended that you deploy the CloudFormation template in its own AWS account.

| AWS Region | AWS CloudFormation Template URL | |
|:-----------|:----------------------------|:-|
| us-east-1 (N. Virginia) |<a href="https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=text2sql&templateURL=" target="_blank">Launch stack</a> ||
| us-west-2 (Oregon) |<a href="https://console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/new?stackName=text2sql&templateURL=" target="_blank">Launch stack</a> |For Llama 3.1|


This CloudFormation template launches a Sagemaker Notebook Instance and an RDS instance running MySQL to execute SQL queries using Llama 3. It also sets up the necessary networking infrastructure, including a Virtual Private Cloud (VPC), subnets, security groups, and flow logs, to facilitate secure communication between the SageMaker notebook and the RDS instance.

1. Click on one of the links above to deploy the solution via CloudFormation in your AWS account. 

2. Click the **Upload** a template file bottom and then upload one of the templates: 1/ [deployment.yaml](cloudformation/text2sql.yaml) 2/ [Llama 3.1 deployment.yaml](cloudformation/text2sql-v2.yaml). Click the orange **Next** button located in the bottom right corner of the console to configure the deployment.

3. Set the stack name as `text2sql` and click **Next** to continue.

5. On the next step, Configure stack options, leave all values as they are and click **Next** to continue.

6. On the Review step

    a. Check the three boxes under Capabilities and transforms to acknowledge the template will create IAM resources and leverage transforms.

    b. Click the Create stack button located at the bottom of the page to deploy the template.

    The stack should take around 10-15 minutes to deploy.

7. Open the generated **SageMakerNotebookURL** Url from the cloudformation outputs above i.e. `https:<sagemaker_notebook_name>.notebook.us-east-1.sagemaker.aws/lab`. 

## Contributing

We welcome community contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
