import os
from pathlib import Path

import smp
import torch
from config import config, Config
from dataset import get_train_val_datasets_cls_big
from torch.utils.data import DataLoader
from trains_utils import get_task_and_logger_from_config, pd_tabl
from utils import make_os_settings, make_logdirs_if_needit
from utils import push_logs


def train_qat(config: Config, train_dataset, valid_dataset, test_dataset, logger):
    train_loader = DataLoader(train_dataset, batch_size=config.bs, shuffle=True, num_workers=config.n_work)
    valid_loader = DataLoader(valid_dataset, batch_size=config.bs, shuffle=False, num_workers=config.n_work)
    test_loader = DataLoader(test_dataset, batch_size=config.bs, shuffle=False, num_workers=config.n_work)

    model = config.model
    model.to(config.device)
    optimizer = config.optimizer(model.parameters(), lr=config.lr)
    scheduler = config.scheduler(optimizer, **config.scheduler_kwargs)

    train_epoch = smp.TrainEpochCustom(
        model=model,
        loss=config.loss,
        metrics=config.metrics,
        optimizer=optimizer,
        device=config.device,
        verbose=True
    )

    valid_tabl = smp.ValidEpochCustomTabl(
        model=model,
        loss=config.loss,
        metrics=config.metrics,
        optimizer=optimizer,
        device=config.device,
        verbose=True
    )

    test_tabl = smp.ValidEpochCustomTabl(
        model=model,
        loss=config.loss,
        metrics=config.metrics,
        optimizer=optimizer,
        device=config.device,
        verbose=True
    )

    experiment_path = os.path.join(config.logs_root, config.experiment_dir)
    qat_models_path = os.path.join(experiment_path, 'qat_models')

    qat_score = 0
    for epoch in range(config.n_epochs):
        train_metrics, model = train_epoch.run(train_loader)


        val_metrics, dict_val = valid_tabl.run(valid_loader)
        logger.report_table("val csv", "remote csv", iteration=0, table_plot=pd_tabl(dict_val))

        test_metrics, dict_test = test_tabl.run(test_loader)
        logger.report_table("test csv", "remote csv", iteration=0, table_plot=pd_tabl(dict_test))

        if val_metrics['acc_document'] > qat_score:
            torch.save(model, os.path.join(qat_models_path, f'{epoch}_{val_metrics["acc_document"]:.4f}.pt'))
            qat_score = val_metrics['acc_document']
            best_model_path = f'{epoch}_{val_metrics["acc_document"]:.4f}.pt'


        scheduler.step(val_metrics['acc_document'])
        #write2log(os.path.join(experiment_path, logname_quant), epoch, val_metrics['acc_document'])

        val_metrics['lr'] = optimizer.param_groups[0]['lr']
        push_logs('train', logger, epoch, train_metrics, config)
        push_logs('val', logger, epoch, val_metrics, config)
        push_logs('test', logger, epoch, test_metrics, config)


    best_model = torch.load(os.path.join(qat_models_path, best_model_path))
    dataset_version = Path(config.dataset_path).parts[-1].replace('.csv', '')
    best_model_name = f'classification_{dataset_version}_{qat_score:.4f}.pt'
    torch.save(best_model, os.path.join(config.model_checkpoints, config.experiment_dir, best_model_name))
    logger.flush()
    return model


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_num
    make_os_settings(config.cuda_num)
    make_logdirs_if_needit(config.logs_root, config.experiment_dir, config.model_checkpoints)
    
    logger, task = get_task_and_logger_from_config(config)

    train_dataset, valid_dataset, test_dataset = get_train_val_datasets_cls_big(config.train_dataset_path, config.val_dataset_path, config.test_dataset_path)

    train_qat(config, train_dataset, valid_dataset,test_dataset, logger)


