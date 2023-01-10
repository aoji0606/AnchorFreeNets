import os
import warnings

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pretrained_models_path = os.path.join(path, 'pretrained_models/')
#s3://model/AnchorFreeNets/pretrained_models
if not os.path.exists(pretrained_models_path):
    warnings.warn("there were no pretrained models, please download the model in s3", ResourceWarning)
    warnings.warn("you have to run the down_pretrained_model.py in AnchorFreeNets directory!!!", ResourceWarning)
    raise ValueError("empty pretrained model dir, you have to run AnchorFreeNets/down_pretrained_model.py!!")