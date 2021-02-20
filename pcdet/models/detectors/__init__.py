from .detector3d_template import Detector3DTemplate
from .LidarMTL import LidarMTL

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'LidarMTL': LidarMTL,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
