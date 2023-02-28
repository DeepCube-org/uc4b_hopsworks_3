import hsml
from hsml.core import dataset_api
from hsml.constants import PREDICTOR_STATE
import argparse

def stopall(ms):
    for deployment in ms.get_deployments(status=PREDICTOR_STATE.STATUS_RUNNING):
        try:
            deployment.stop()
        except:
            deployment.get_logs()
def lastversion(ms, model_name, status = PREDICTOR_STATE.STATUS_RUNNING):
    deployments = ms.get_deployments(status=status)
    deployments = [deployment for deployment in deployments if deployment.model_name == model_name]
    deployments = sorted(deployments, key=lambda x: x.model_version, reverse=True)
    deployment = deployments[0]
    return(deployment)


MODEL_NAME = 'modelserving'
if(__name__=='__main__'):

    parser = argparse.ArgumentParser(description = 'Push to Hopsworks from github')
    parser.add_argument('--api_key_value', required=False, type=str, help='HopsWorks API key', default = None) 
    parser.add_argument('--version', required=True, type=int, help='Version of the model') 
    args = parser.parse_args()
    
    if(args.api_key_value is None):
        with open('hopsworks.key') as f:
            args.api_key_value = f.readline()

    
    MODEL_VERSION = args.version
    DEPLOYMENT_NAME = MODEL_NAME+str(MODEL_VERSION)

    # Create a connection
    connection = hsml.connection(
        host = 'hopsworks.deepcube-h2020.gael-systems.com',
        project = 'test_test_2',
        port = 8181,
        api_key_value = args.api_key_value
    )


    script_file = 'Resources/{MODEL_NAME}/{VERSION}/'.format(MODEL_NAME=MODEL_NAME, VERSION=MODEL_VERSION)
    dataset_api = dataset_api.DatasetApi()

    # Get the model registry handle for the project's model registry
    mr = connection.get_model_registry()

    # Get the model serving handle for the current model registry
    ms = connection.get_model_serving()

    #model = sklearn.neighbors.KNeighborsClassifier()
    #joblib.dump(model, 'model.pkl')

    model = mr.python.create_model(name=MODEL_NAME, version=MODEL_VERSION)
    model.save('model.pkl')

    model = mr.get_model(MODEL_NAME, version=MODEL_VERSION)

    dataset_api.upload('predictor.py', script_file)

    predictor = ms.create_predictor(model, model_server="PYTHON", serving_tool="KSERVE", script_file='/Projects/test_test_2/'+script_file+'predictor.py')

    deployment = ms.create_deployment(predictor, name=DEPLOYMENT_NAME)

    try:
        deployment.save()
        stopall(ms)
        deployment.start()

    except Exception as e:
        deployment.get_logs()
