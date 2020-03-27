from .BN_Inception import BN_Inception
from .Inception import inception_v1_ml
from .resnet import resnet_all

__factory = {
    'BN-Inception': BN_Inception,
    "Inception": inception_v1_ml,
    "ResNet": resnet_all
}

def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    
    if "ResNet" in name:
        n = name[6:]
        return __factory[name[:6]](which_resnet=n, *args, **kwargs)
    else:
        if name not in __factory:
            raise KeyError("Unknown network:", name)
        return __factory[name](*args, **kwargs)
