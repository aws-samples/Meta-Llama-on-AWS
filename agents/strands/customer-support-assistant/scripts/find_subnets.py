import boto3

# Create a Boto3 session
my_session = boto3.session.Session()

# Get the region name from the session
current_region = my_session.region_name

ec2 = boto3.client('ec2', region_name=current_region)
instance_type = 'g6e.4xlarge'

# Find supported AZs
offerings = ec2.describe_instance_type_offerings(
    LocationType='availability-zone',
    Filters=[{'Name': 'instance-type', 'Values': [instance_type]}]
)
zones = [off['Location'] for off in offerings['InstanceTypeOfferings']]

# Find subnets in those AZs
subnets_resp = ec2.describe_subnets(
    Filters=[{'Name': 'availabilityZone', 'Values': zones}]
)
subnet_ids = [sn['SubnetId'] for sn in subnets_resp['Subnets']]
#print("Compatible subnets:", " ".join(subnet_ids))
out = ",".join(subnet_ids)
print(f"\"{out}\"")