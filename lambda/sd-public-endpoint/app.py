import os
from urllib.parse import unquote

import boto3
from botocore.config import Config
from chalice import Chalice, Response
from sagemaker.session import Session
from essential_generators import DocumentGenerator
from sagemaker.huggingface.model import HuggingFacePredictor, HuggingFaceModel
import io
import time
import json
import base64
import sys
from pathlib import Path
import uuid

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
CLASS_NAME_FOR_TRAINING = "sks" # this is the class name we'll be giving. Use this in prompt to refer to the trained object
CLASS_NAME_FOR_PARENT_OBJECT = "dog" # class name for the parent object. Default will be dog, e.g. "a photo of an (((sks dog)))" would create a photo of our custom trained dog

BASE_RESOURCE_NAME = "huggingface-pytorch-inference" # we'll add the model id to the end when creating and deleting
INSTANCE_TYPE = "ml.g4dn.xlarge" # instance type for the endpoint aka the type of GPU

PASSWORD = "asdoaj023199jdoacalksdafafnav0239q4raslkdj" # password to use for the API, firebase will use this to authenticate

# get sagemaker role
sagemaker_role = ""
CURR_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
with open(CURR_PATH / ".." / ".." / "terraform" / "sagemaker-role-arn.txt", "r") as f:
    sagemaker_role = f.read()


config = Config(read_timeout=30, retries={"max_attempts": 0})
sagemaker_runtime_client = boto3.client("sagemaker-runtime", config=config)
sagemaker_session = Session(sagemaker_runtime_client=sagemaker_runtime_client)
# predictor = HuggingFacePredictor(
    # endpoint_name=ENDPOINT_NAME, sagemaker_session=sagemaker_session
# )
# sentence_generator = DocumentGenerator()

app = Chalice(app_name="PawPrintsAI")
# function to validate password for current request
def validate_password():
    if app.current_request.headers.get("password") != PASSWORD:
        return False
    return True

@app.route("/")
def index():
    return "Server is running."

@app.route("/get_status/{model_id}")
def get_status(model_id):
    job = sagemaker_session.sagemaker_client.describe_training_job(
        TrainingJobName=model_id
    )
    print("Fetched status for " + model_id)
    return job

@app.route("/list_training_jobs_with_status/{status}")
def list_training_jobs(status):
    jobs = sagemaker_session.sagemaker_client.list_training_jobs(
        SortBy="CreationTime", SortOrder="Descending", StatusEquals=status
    )
    job_names = map(lambda job: [job['TrainingJobName'], job['TrainingJobStatus']],  jobs['TrainingJobSummaries'])
    print("Fetched " + str(len(job_names)) + " jobs with status " + status)
    return {"jobs": list(job_names)}


# check if endpoint exists
@app.route("/check_endpoint/{model_id}", cors=True)
def check_endpoint(model_id):
    if not validate_password():
        return {"status": "error", "message": "Invalid Access"}
    # return true or false if endpoint exists
    model_id = unquote(model_id)
    endpoint_name = BASE_RESOURCE_NAME + "-" + model_id
    try:
        sagemaker_session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        return {"exists": True}
    except:
        return {"exists": False}
    

@app.route("/create_endpoint/{model_id}", cors=True)
def create_endpoint(model_id):
    # if not validate_password():
    #     return {"status": "error", "message": "Invalid Access"}
    starttime = time.time()
    # in order to generate art, we need to create an endpoint for the model
    model_id = unquote(model_id)
    # check if model exists without downloading, if it does return error since we shouldn't create a second endpoint
    try:
        endpoint_name = BASE_RESOURCE_NAME + "-" + model_id
        sagemaker_session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        error_id = str(uuid.uuid4())
        print("Checkpoint already exists, returning error. Error ID: " + error_id)
        return {"status": "error", "message": "Checkpoint already exists", "error_id": error_id}
    except:
        print("Checked Checkpoint isn't already running, continuing...")
    
    MODEL_PARAMS = {
        "model_data": "s3://pawprintsai-models/" + model_id + ".tar.gz",
        "name": BASE_RESOURCE_NAME + "-" + model_id,
        "role": sagemaker_role,
        "transformers_version": "4.12",
        "pytorch_version": "1.9",
        "py_version": "py38",
    }
    # verify model exists without downloading
    s3 = boto3.resource("s3")
    try:
        s3.meta.client.head_object(Bucket="pawprintsai-models", Key=model_id + ".tar.gz")
    except Exception as e:
        print(e)
        error_id = str(uuid.uuid4())
        print("Model doesn't exist, returning error. Error ID: " + error_id)
        return {"status": "error", "message": "Model doesn't exist", "error_id": error_id}
    
    huggingface_model = HuggingFaceModel(**MODEL_PARAMS)

    # deploy the endpoint endpoint
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=BASE_RESOURCE_NAME + "-" + model_id,
    )

    print(f"[1;32mCreated endpoint in: {time.time() - starttime}s[0m")
    return {"status": "success"}

@app.route("/delete_endpoint/{model_id}", cors=True)
def delete_endpoint(model_id):
    if not validate_password():
        return {"status": "error", "message": "Invalid Access"}
    
    # once we've generated all the art for a specific order, we want to delete the endpoint to avoid incurring costs
    model_id = unquote(model_id)
    # check if model exists without downloading
    s3 = boto3.resource("s3")
    try:
        s3.meta.client.head_object(Bucket="pawprintsai-models", Key=model_id + ".tar.gz")
    except:
        error_id = str(uuid.uuid4())
        print(f"[1;31m[{error_id}] Model not found for model {model_id}[0m")
        return {"error": "Model not found", "error_id": error_id}
    sagemaker_session = boto3.client("sagemaker")
    predictor_delete_ok = False
    MODEL_PARAMS = {
        "model_data": "s3://pawprintsai-models/" + model_id + ".tar.gz",
        "name": BASE_RESOURCE_NAME + "-" + model_id,
        "role": sagemaker_role,
        "transformers_version": "4.12",
        "pytorch_version": "1.9",
        "py_version": "py38",
    }
    try:
        predictor = HuggingFacePredictor(endpoint_name=BASE_RESOURCE_NAME + "-" + model_id)
        predictor.delete_endpoint()
        predictor_delete_ok = True
    except Exception as e:
        print(
            f"Predictor deletion failed: {e} "
            "Trying to delete the model and endpoint configuration directly",
            file=sys.stderr,
        )

    if predictor_delete_ok:
        print("[1;32mPredictor deleted successfully[0m")
        return {"status": "success"}

    try:
        model = HuggingFaceModel(**MODEL_PARAMS)
        model.delete_model()
    except Exception as e:
        print(f"[1;31mModel deletion failed[0m: {e}", file=sys.stderr)

    try:
        sagemaker_session.delete_endpoint_config(EndpointConfigName=BASE_RESOURCE_NAME + "-" + model_id)
        print("[1;32mEndpoint config deleted successfully[0m")
        return {"status": "success"}
    except Exception as e:
        print(f"[1;31mEndpoint config deletion failed[0m: {e}", file=sys.stderr)
    print("[1;31mResponse: Endpoint deletion failed[0m")
    return {"error": "Endpoint deletion failed"}

# path with {prompt} and {model_id} as inputs
@app.route("/run_model/{model_id}", methods=["POST"], cors=True)
def run_model(model_id):
    if not validate_password():
        return {"status": "error", "message": "Invalid Access"}
    
    model_id = unquote(model_id)
    # prompt = unquote(prompt)
    body = app.current_request.json_body
    if "prompt" not in body or prompt == "":
        error_id = str(uuid.uuid4())
        print(f"[1;31m[{error_id}] No prompt provided[0m")
        print(f"[1;31m[{error_id}] Body: {body}[0m")
        return {"error": "No prompt provided", "error_id": error_id}
    prompt = body["prompt"]
    negative_prompt = body["negative_prompt"]
    # check that endpoint is running, if not start it
    sagemaker_session = boto3.client("sagemaker")
    try:
        sagemaker_session.describe_endpoint(EndpointName=BASE_RESOURCE_NAME + "-" + model_id)
    except:
        print(f"[1;31mEndpoint not found, creating endpoint for model {model_id}[0m")
        res = create_endpoint(model_id)
        if res["status"] == "error":
            return res
        
    # load the model
    starttime = time.time()
    predictor = HuggingFacePredictor(endpoint_name=BASE_RESOURCE_NAME + "-" + model_id)
    
    g_cuda = torch.Generator(device='cuda')
    seed = 47853 #@param {type:"number"}, will make things more reproducible
    g_cuda.manual_seed(seed)

    # default running settings
    run_settings = {
        "prompt": prompt,
        "height": 768,
        "width": 768,
        "num_images_per_prompt": 1,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "generator": g_cuda,
    }
    # check if we have a negative prompt
    if negative_prompt in body and body[negative_prompt] != "":
        run_settings["negative_prompt"] = body[negative_prompt]
    # run using run_settings
    res = predictor.predict(data=run_settings)
    image_bytes = base64.b64decode(res["data"])
    print(f"[1;32m Ran model in: {time.time() - starttime}s[0m")
    # return image output
    return {"image": image_bytes}
     
# path with {model_id} and an array of images as inputs
@app.route("/train_model/{model_id}", methods=["POST"], cors=True)
def train_model(model_id):
    if not validate_password():
        return {"status": "error", "message": "Invalid Access"}
    model_id = unquote(model_id)
    # verify the model training job isn't already running
    try:
        job = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=model_id
        )
        if job["TrainingJobStatus"] == "InProgress":
            error_id = str(uuid.uuid4())
            print(f"[1;31m[{error_id}] Model training already in progress[0m")
            return {"error": "Model training already in progress", "error_id": error_id}
    except:
        pass
    # get images from request
    images = app.current_request.json_body["images"]

    try:

        s3 = boto3.resource("s3")
        bucket = s3.Bucket("pawprintsai-models")
        for i, image in enumerate(images):
            # upload files to bucket
            # image name format example: sks (0).jpg, this'll help with dreambooth training
            bucket.upload_fileobj(
                io.BytesIO(base64.b64decode(image["image"])),
                f"images/{model_id}/{CLASS_NAME_FOR_TRAINING} ({i}).jpg",
            )
    except Exception as e:
        # create random ID for error
        error_id = str(uuid.uuid4())
        print(f"[1;31m[{error_id}] Error uploading images:[0m {e}")
        return {"error": "Error uploading images", "error_id": error_id}
    

    print("[1;32mUploaded images[0m")

    # start running pawprints_dreambooth_real.ipynb in a training job on sagemaker
    # this'll take a while, so we'll return a 202 and the client can poll the status
    # of the training job
    #
    # we'll use the same model_id as the training job name
    # this'll make it easier to track the status of the training job
    # and also make it easier to delete the training job
    # if the user wants to delete the model
    #

    # try catch any errors
    try:
        # create a lambda client
        lambda_client = boto3.client("lambda")
        # call the lambda function
        lambda_client.invoke(
            FunctionName="pawprintsai-training",
            InvocationType="Event",
            Payload=json.dumps(
                {
                    "model_id": model_id,
                }
            ),
        )
    except Exception as e:
        error_id = str(uuid.uuid4())
        print(f"[1;31m[{error_id}] Failed to start training job[0m: {e}", file=sys.stderr)
        return {"error": "Failed to start training job", "error_id": error_id}
    
    
    print("[1;32mStarted training job[0m")
    return {"status": "training_started"}

