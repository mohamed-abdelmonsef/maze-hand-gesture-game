from fastapi.testclient import TestClient
from app import app  # replace 'main' with your actual filename if different
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = TestClient(app)


def test_predict_endpoint():
    logger.info("Testing /predict endpoint")


    sample_flat_landmarks = [0.3616105318069458, 0.7152789235115051, 5.057307816969114e-7, 0.3982469141483307, 0.6665682792663574,
                            -0.01809690147638321, 0.4151521325111389, 0.5815799236297607, -0.02639557421207428, 0.39411088824272156,
                            0.5151383280754089, -0.0351032130420208, 0.3663700222969055, 0.4757891297340393, -0.041851215064525604, 
                            0.4014818072319031, 0.4772266447544098, -0.013069983571767807, 0.4220424294471741, 0.38472792506217957, 
                            -0.029957016929984093, 0.4343414008617401, 0.3309529423713684, -0.039192914962768555, 0.4433809220790863, 
                            0.28655874729156494, -0.04318036511540413, 0.36900651454925537, 0.4749997854232788, -0.01621393859386444, 
                            0.37042585015296936, 0.36430823802948, -0.036023396998643875, 0.37255722284317017, 0.2962542176246643, 
                            -0.046486202627420425, 0.37289881706237793, 0.24529731273651123, -0.049383070319890976, 0.3405744433403015, 
                            0.49769148230552673, -0.021658344194293022, 0.3576485514640808, 0.4602510929107666, -0.05245034396648407, 
                            0.3748930096626282, 0.5155668258666992, -0.05773286521434784, 0.38085034489631653, 0.5629295706748962, 
                            -0.05074629932641983, 0.3189316689968109, 0.5344305038452148, -0.027997370809316635, 0.34685638546943665, 
                            0.5150024890899658, -0.05156423896551132, 0.36842721700668335, 0.5570330619812012, -0.05125751718878746, 
                            0.3783669173717499, 0.5969078540802002, -0.04365749657154083]   # peace gesture

    request_payload = {
        "data": [sample_flat_landmarks]  
    }

    response = client.post("/predict", json=request_payload)
    assert response.status_code == 200, f"Failed with: {response.text}"
    
    response_json = response.json()
    
    assert "prediction" in response_json, "'prediction' key missing in response"
    assert "command" in response_json, "'command' key missing in response"
    assert "message" in response_json
    assert isinstance(response_json["prediction"], str)
    assert isinstance(response_json["command"], str)
    assert response_json["message"] == "Prediction successful"
