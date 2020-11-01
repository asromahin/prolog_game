import typing as tp
from dataclasses import asdict, dataclass

from adam_gcc import Adam_GCC2
from losses import FocalLoss
from metrics import Accuracy, Fscore
from mobimet_v2 import MobileNetV2
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

rlron_pl_kwargs = {
    'patience': 3,
    'factor': 0.5,
    'mode': 'max',
    'verbose': True,
}


@dataclass
class Config:
    size: int
    seed: int
    device: str
    cuda_num: str
    loss: nn.Module
    optimizer: type(Optimizer)
    scheduler: type(_LRScheduler)
    scheduler_kwargs: tp.Dict[str, tp.Any]
    log_name: str
    metrics: tp.List[tp.Any]
    n_epochs: int
    bs: int
    n_work: int
    lr: float
    logs_root: str
    experiment_dir: str
    model: nn.Module
    dataset_path: str
    train_dataset_path: str
    val_dataset_path: str
    test_dataset_path: str
    model_checkpoints: str
    use_resized_images: bool
        
    def to_dict(self):
        return asdict(self)



config = Config(
    size=512,
    seed=69,
    device='cuda',
    cuda_num='1',
    loss=FocalLoss(),
    optimizer=Adam_GCC2,
    scheduler=ReduceLROnPlateau,
    scheduler_kwargs=rlron_pl_kwargs,
    log_name='opa',
    metrics=[
        Accuracy(activation='sigmoid'),
        Fscore(activation='sigmoid')
    ],
    n_epochs=70,
    bs=35,
    n_work=4,
    lr=1e-3,
    logs_root='experiments',
    experiment_dir='mobilenet_v2_titul',
    model=MobileNetV2(),
    dataset_path='/data/hackathon/data_full.csv',
    train_dataset_path='/data/hackathon/data_train.csv',
    val_dataset_path='/data/hackathon/data_val.csv',
    test_dataset_path='/data/hackathon/data_test.csv',
    model_checkpoints='/data/checkpoints/Hackathon/',
    use_resized_images=True,
)
