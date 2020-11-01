import typing as tp

from torch import nn
from trains import Logger, Task

from src.classification.model_train.config import Config


def get_task_and_logger_from_config(config: Config) -> tp.Tuple[Logger, Task]:
    task = Task.init(project_name='[Hackathon]CNN', task_name=config.experiment_dir)
    logger = task.get_logger()
    task.connect(_config2trains_hyperparams(config))
    return logger, task


def _config2trains_hyperparams(config: Config) -> dict:
    res = {}
    for k, v in config.to_dict().items():
        if isinstance(v, nn.Module):
            res[k] = v.__class__.__name__
        else:
            res[k] = str(v)
    return res

import pandas as pd

def pd_tabl(dict_value):

    def mean_to_key(dict_value, key):
        return [round(dict_value[x][key].value()[0], 6) for x in dict_value.keys()]

    def quantity_to_key(dict_value, key):
        return [dict_value[x][key].n for x in dict_value.keys()]

    df = pd.DataFrame({'Документ': [x for x in dict_value.keys()],
                       'clf_document': mean_to_key(dict_value, 'clf_document'),
                       'Fscore': mean_to_key(dict_value, 'Fscore'),
                        'quantity': quantity_to_key(dict_value, 'clf_document'),
                       })

    return df.dropna()