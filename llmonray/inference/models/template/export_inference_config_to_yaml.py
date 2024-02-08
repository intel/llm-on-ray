import yaml
import os
from llmonray.inference.inference_config import InferenceConfig

ic = InferenceConfig()

with open(os.path.dirname(__file__) + "/inference_config_template.yaml", "w") as f:
    yaml.dump(ic.dict(), f, sort_keys=False)
