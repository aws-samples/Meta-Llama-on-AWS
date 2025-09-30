import boto3
from typing import List

def get_ollama_ip(tag_name: str = 'ollama_customer_support') -> List[str]:
    """
    Filter EC2 instances based on tag name and running state.
    
    Args:
        tag_name (str): The tag name to filter instances (default: 'ollama_customer_support')
        
    Returns:
        List[str]: List of public IP addresses of matching instances
    """
    try:
        # Initialize EC2 client
        ec2_client = boto3.client('ec2')
        
        # Define filters
        filters = [
            {
                'Name': 'tag:Name',
                'Values': [tag_name]
            },
            {
                'Name': 'instance-state-name',
                'Values': ['running']
            }
        ]
        
        # Describe instances with filters
        response = ec2_client.describe_instances(Filters=filters)
        
        # Initialize list to store public IPs
        public_ips = []
        
        # Extract public IPs from the response
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                if 'PublicIpAddress' in instance:
                    public_ips.append(instance['PublicIpAddress'])
        
        return public_ips
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return []

# Example usage
if __name__ == "__main__":
    # Get instances with default tag name
    ips = get_ollama_ip()
    
    if ips:
        print("Found instances with public IPs:")
        for ip in ips:
            print(f"{ip}")
    else:
        print("No matching instances found or no public IPs available")
