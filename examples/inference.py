import hsml
import numpy as np
from create_model import MODEL_NAME, lastversion

host = 'c.app.hopsworks.ai'
port = 443
project = 'test_test'


api_key_value = None
if(api_key_value is None):
    with open('hopsworks.key') as f:
        api_key_value = f.readline()

# Create a connection
connection = hsml.connection(
    host = host,
    project = project,
    port = port,
    api_key_value = api_key_value
)

# get Hopsworks Model Serving
ms = connection.get_model_serving()



# get deployment object
deployment = lastversion(ms, model_name=MODEL_NAME)

# get Hopsworks Model Registry
mr = connection.get_model_registry(project=project)

# get model
model = mr.get_model(deployment.model_name, deployment.model_version)

attributes_height_kernel    = np.ones((100, 1, 80, 80)).tolist(),
attributes_amplitude_kernel = np.ones((100, 1, 80, 80)).tolist(),
attributes_surf_kernel      = np.ones((100, 1, 80, 80)).tolist(),
attributes_B4_kernel        =  np.ones((100, 1, 80, 80)).tolist(),
attributes_B3_kernel        =  np.ones((100, 1, 80, 80)).tolist(),
attributes_B2_kernel        =  np.ones((100, 1, 80, 80)).tolist(),
attributes_B8_kernel        =  np.ones((100, 1, 80, 80)).tolist(),
attributes_B11_kernel       =  np.ones((100, 1, 80, 80)).tolist(),
attributes_B12_kernel       =  np.ones((100, 1, 80, 80)).tolist(),
attributes_velocity         =  np.ones((100, 1)).tolist(),
attributes_coherence        =  np.ones((100, 1)).tolist(),
attributes_mir              =  np.ones((100, 1)).tolist()


print('MODEL NAME:', deployment.model_name, 'MODEL VERSION:', deployment.model_version)
try:
    outputs = deployment.predict({"instances": [
        {
            'attributes_height_kernel'    : attributes_height_kernel,
            'attributes_amplitude_kernel' : attributes_amplitude_kernel,
            'attributes_surf_kernel'      : attributes_surf_kernel,
            'attributes_B4_kernel':         attributes_B4_kernel,
            'attributes_B3_kernel':         attributes_B3_kernel,
            'attributes_B2_kernel':         attributes_B2_kernel,
            'attributes_B8_kernel':         attributes_B8_kernel,
            'attributes_B11_kernel':        attributes_B11_kernel,
            'attributes_B12_kernel':        attributes_B12_kernel,
            'attributes_velocity':          attributes_velocity,
            'attributes_coherence':         attributes_coherence,
            'attributes_mir':               attributes_mir
        }
    ]})
    for output in outputs['predictions']:
        output = np.array(output)
        print(output.shape)

except Exception as e:
    deployment.get_logs()


