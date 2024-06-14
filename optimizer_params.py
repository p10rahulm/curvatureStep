from optimizers.simplesgd import SimpleSGD
from optimizers.simplesgd_curvature import SimpleSGDCurvature
from optimizers.adam import Adam
from optimizers.adam_curvature import AdamCurvature
from optimizers.heavyball import HeavyBall
from optimizers.heavyball_curvature import HeavyBallCurvature
from optimizers.nag import NAG
from optimizers.nag_curvature import NAGCurvature
from optimizers.adagrad import Adagrad
from optimizers.adagrad_curvature import AdagradCurvature
from optimizers.adadelta import Adadelta
from optimizers.adadelta_curvature import AdadeltaCurvature
from optimizers.rmsprop import RMSProp
from optimizers.rmsprop_curvature import RMSPropCurvature
from optimizers.rmsprop_with_momentum import RMSPropMomentum
from optimizers.rmsprop_with_momentum_curvature import RMSPropMomentumCurvature
from optimizers.adamw import AdamW
from optimizers.adamw_curvature import AdamWCurvature
from optimizers.nadam import NAdam
from optimizers.nadam_curvature import NAdamCurvature
from optimizers.nadamw import NAdamW
from optimizers.nadamw_curvature import NAdamWCurvature
from optimizers.amsgrad import AMSGrad
from optimizers.amsgrad_curvature import AMSGradCurvature
from optimizers.shampoo import Shampoo
from optimizers.shampoo_curvature import ShampooCurvature

optimizers = [
    (SimpleSGD, {'lr': 1e-3}),
    (SimpleSGDCurvature, {'lr': 1e-3, 'epsilon': 0.01}),
    (Adam, {'lr': 1e-3, 'betas': (0.9, 0.999)}),
    # (AdamCurvature, {'lr': 1e-3, 'betas': (0.9, 0.999), 'epsilon': 0.01}),
    (HeavyBall, {'lr': 1e-3, 'momentum': 0.95}),
    (HeavyBallCurvature, {'lr': 1e-3, 'momentum': 0.55, 'epsilon': 0.01}),
    (NAG, {'lr': 1e-3, 'momentum': 0.95}),
    (NAGCurvature, {'lr': 1e-3, 'momentum': 0.55, 'epsilon': 0.01}),
    (Adagrad, {'lr': 1e-2, 'eps': 1e-10}),
    # (AdagradCurvature, {'lr': 1e-2, 'eps': 1e-10, 'epsilon': 0.01}),
    (Adadelta, {'lr': 1.0, 'rho': 0.9, 'eps': 1e-6}),
    # (AdadeltaCurvature, {'lr': 1.0, 'rho': 0.6, 'eps': 1e-6, 'epsilon': 0.01}),
    (RMSProp, {'lr': 1e-2, 'alpha': 0.9, 'eps': 1e-8, 'weight_decay': 0}),
    # (RMSPropCurvature, {'lr': 1e-2, 'alpha': 0.99, 'eps': 1e-8, 'weight_decay': 0, 'epsilon': 0.01}),
    (RMSPropMomentum, {'lr': 1e-2, 'alpha': 0.9, 'eps': 1e-8, 'weight_decay': 0, 'momentum': 0.05}),
    # (RMSPropMomentumCurvature, {'lr': 1e-2, 'alpha': 0.99, 'eps': 1e-8, 'weight_decay': 0, 'momentum': 0.1, 'epsilon': 0.01}),
    (AdamW, {'lr': 1e-3, 'betas': (0.9, 0.999), 'weight_decay': 0.01}),
    # (AdamWCurvature, {'lr': 1e-3, 'betas': (0.9, 0.999), 'epsilon': 0.01}),
    (NAdam, {'lr': 1e-3, 'betas': (0.9, 0.999)}),
    # (NAdamCurvature, {'lr': 1e-3, 'betas': (0.9, 0.999), 'epsilon': 0.01}),
    (NAdamW, {'lr': 1e-3, 'betas': (0.9, 0.999), 'weight_decay': 0.01}),
    # (NAdamWCurvature, {'lr': 1e-3, 'betas': (0.9, 0.999), 'epsilon': 0.01}),
    (AMSGrad, {'lr': 1e-3, 'betas': (0.9, 0.999)}),
    # (AMSGradCurvature, {'lr': 1e-3, 'betas': (0.9, 0.999), 'epsilon': 0.01}),
    (Shampoo, {'lr': 1e-3, 'momentum': 0.1}),
    # (ShampooCurvature, {'lr': 1e-3, 'momentum': 0.1, 'epsilon': 0.01}),
]
