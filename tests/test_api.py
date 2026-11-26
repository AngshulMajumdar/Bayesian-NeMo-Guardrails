from fastapi.testclient import TestClient
from bayesian_nemo.api import app

client=TestClient(app)

def test_health_and_config():
    assert client.get('/health').status_code==200
    assert client.get('/config').json()['phases']['input']['example_data_only']


def test_api_decisions():
    for risk,expected in ((.02,'allow'),(.5,'review'),(.99,'block')):
        r=client.post('/evaluate',json={'phase':'input','scores':{'injection_risk':risk,'toxicity_risk':.02}})
        assert r.status_code==200
        assert r.json()['decision']==expected


def test_api_input_validation():
    assert client.post('/evaluate',json={'phase':'missing','scores':{}}).status_code==400
    assert client.post('/evaluate',json={'phase':'input','scores':{'injection_risk':.02}}).json()['decision']=='review'
