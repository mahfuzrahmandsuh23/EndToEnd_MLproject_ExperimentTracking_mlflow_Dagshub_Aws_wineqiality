# EndToEnd_MLproject_ExperimentTracking_mlflow_Dagshub_Aws_wineqiality
# MLflow-Basic-Demo



## For current Dagshub:
#Initising Dagshub

import dagshub
dagshub.init(repo_owner='mahfuzrahmandsuh23', repo_name='EndToEnd_MLproject_ExperimentTracking_mlflow_Dagshub_Aws_wineqiality', mlflow=True)

import mlflow
with mlflow.start_run():
mlflow.log_param('parameter name', 'value')
mlflow.log_metric('metric name', 1)






# MLflow on AWS

## MLflow on AWS Setup:

1. Login to AWS console.
2. Create IAM user with AdministratorAccess
3. Export the credentials in your AWS CLI by running "aws configure"
4. Create a s3 bucket
5. Create EC2 machine (Ubuntu) & add Security groups 5000 port

Run the following command on EC2 machine
```bash
sudo apt update

sudo apt install python3-pip

sudo apt install pipenv

sudo apt install virtualenv

mkdir mlflow

cd mlflow

pipenv install mlflow

pipenv install awscli

pipenv install boto3

pipenv shell


## Then set aws credentials
aws configure


#Finally 
mlflow server -h 0.0.0.0 --default-artifact-root s3://mlflowbucket25

#open Public IPv4 DNS to the port 5000


#set uri in your local terminal and in your code 
export MLFLOW_TRACKING_URI=http://ec2-51-21-202-83.eu-north-1.compute.amazonaws.com:5000/
```


