import hsfs

api_key_value = None
if(api_key_value is None):
    with open('hopsworks.key') as f:
        api_key_value = f.readline()

conn = hsfs.connection(
    host = 'c.app.hopsworks.ai',
    project = 'test_test',
    port = 443,
    api_key_value = api_key_value,
    hostname_verification=True # Disable for self-signed certificates
)         


fs = conn.get_feature_store(name='test_test_featurestore')
td = fs.get_feature_view('test', version=1)

transactions_to_score = td.get_feature_vectors()