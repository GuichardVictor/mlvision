from .cnn import SimpleCnn
import torchvision.models

MODEL_REGISTRY = {
    "simple_cnn": SimpleCnn,
    "resnet50": torchvision.models.resnet50,
    "resnet18": torchvision.models.resnet18,
}

def build_model(architecture, kargs):
    model_class = MODEL_REGISTRY[architecture]

    return model_class(**kargs)