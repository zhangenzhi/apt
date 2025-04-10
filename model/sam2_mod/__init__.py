from hydra import initialize_config_module
import sys
sys.path.append("./")
initialize_config_module("model.sam2_mod.configs", version_base="1.2")