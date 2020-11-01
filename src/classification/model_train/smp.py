import sys

import numpy as np
import torch
from tqdm import tqdm as tqdm


# import apex


class Meter(object):
    '''Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    '''

    def reset(self):
        '''Resets the meter to default settings.'''
        pass

    def add(self, value):
        '''Log a new value to the meter
        Args:
            value: Next restult to include.
        '''
        pass

    def value(self):
        '''Get the value of the meter in the current state.'''
        pass


class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k[0:6], v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                   # print(metric_fn(y_pred, y))
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True, use_apex=False):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.use_apex = use_apex

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        if self.use_apex:
            print('')
            # with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction




class TrainEpochCustom(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True, cls_w_1=1, cls_w_2=2, seg_w=1, use_distil=False):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,

        )
        self.cls_w_1 = cls_w_1
        self.cls_w_2 = cls_w_2
        self.seg_w = seg_w
        self.optimizer = optimizer
        #self.loss_clf = FocalLoss() # nn.BCEWithLogitsLoss()
        self.loss_clf = loss

        self.use_distil = use_distil

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics if metric.__name__ != 'accuracy'}
        metrics_meters['acc_document'] = AverageValueMeter()
        metrics_meters['acc_titul'] = AverageValueMeter()


        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not self.verbose) as iterator:
            for x, y, filename, titul in iterator:
                x, y, titul = x.to(self.device), y.to(self.device), titul.to(self.device)

                loss, y_pred_clf_1, titul_pred = self.batch_update(x, y, titul)
                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    if metric_fn.__name__ == 'Accuracy':
                        metric_value = metric_fn(y_pred_clf_1, y).cpu().detach().numpy()
                        metrics_meters['acc_document'].add(metric_value)
                        metric_value = metric_fn(titul_pred, titul).cpu().detach().numpy()
                        metrics_meters['acc_titul'].add(metric_value)
                    else:
                        metric_value = metric_fn(y_pred_clf_1, y).cpu().detach().numpy()
                        metrics_meters[metric_fn.__name__].add(metric_value)


                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs, self.model

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y_clf_1, titul):
        self.optimizer.zero_grad()
        prediction_clf_1, titul_pred = self.model.forward(x)
        loss = self.loss_clf(prediction_clf_1,y_clf_1) * self.cls_w_1 + self.loss_clf(prediction_clf_1,y_clf_1) * self.cls_w_2
        loss.backward()
        self.optimizer.step()
        return loss, prediction_clf_1, titul_pred


from constants import DOCNAME2CLASS, BACKDOCNAME2CLASS


def dict_metric():
    def metric_m():
        metrics_meters = {'Fscore': AverageValueMeter(),
                          'acc_document': AverageValueMeter(),
                          'acc_titul': AverageValueMeter(),
                           'clf_quality': AverageValueMeter(),}
        return metrics_meters

    keys_dict = list(DOCNAME2CLASS.keys())
    metrics_all = [metric_m() for _ in range(len(DOCNAME2CLASS.keys()))]
    dict_value = dict(zip(keys_dict, metrics_all))
    return dict_value


class ValidEpochCustomTabl(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True, cls_w_1=1, cls_w_2=2, seg_w=1,
                 use_distil=False):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='val_tabl',
            device=device,
            verbose=verbose,
        )
        self.b = torch.Tensor(list(range(0, len(BACKDOCNAME2CLASS))))
        self.cls_w_1 = cls_w_1
        self.cls_w_2 = cls_w_2
        self.seg_w = seg_w
        self.optimizer = optimizer
        self.loss_clf = loss
        self.use_distil = use_distil


    def run(self, valid_loader):
        device = self.device
        self.on_epoch_start()

        dict_value = dict_metric()
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics if metric.__name__ != 'accuracy'}
        metrics_meters['acc_document'] = AverageValueMeter()
        metrics_meters['acc_titul'] = AverageValueMeter()


        with tqdm(valid_loader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for imgs, y_all, filename, titul in iterator:
                imgs, y_class, titul= imgs.to(device), y_all.to(device), titul.to(device)
                loss, preds_clf_1, titul_pred = self.batch_update(imgs, y_class, titul)
                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                metrics_meters, metric_value = self.update_batch_metrics(metrics_meters,
                                                                         y_class, preds_clf_1,
                                                                         titul, titul_pred)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                for pred_clf_1, y_clas, tit_pred, tit in zip(preds_clf_1, y_class, titul_pred, titul):
                    # update table metrics
                    class_str = self.class_to_str(y_clas.cpu().detach())

                    for metric_fn in self.metrics:
                        if metric_fn.__name__ == 'Accuracy':
                            metric_value = metric_fn(pred_clf_1.unsqueeze(0), y_clas.unsqueeze(0)).cpu().detach().numpy()
                            #metrics_meters['acc_document'].add(metric_value)
                            dict_value[class_str]['acc_document'].add(metric_value)
                            metric_value = metric_fn(tit_pred.unsqueeze(0), tit.unsqueeze(0)).cpu().detach().numpy()
                            dict_value[class_str]['acc_titul'].add(metric_value)
                            #metrics_meters['acc_titul'].add(metric_value)
                        else:
                            metric_value = metric_fn(pred_clf_1.unsqueeze(0), y_clas.unsqueeze(0)).cpu().detach().numpy()
                            dict_value[class_str][metric_fn.__name__].add(metric_value)
                            #metrics_meters[metric_fn.__name__].add(metric_value)

                    if self.verbose:
                        s = self._format_logs(logs)
                        iterator.set_postfix_str(s)
        return logs, dict_value

    def on_epoch_start(self):
        self.model.eval()

    def class_to_str(self, index_list):
        name = [BACKDOCNAME2CLASS[int(self.b @ index_list)]]
        return name[0]

    def update_batch_metrics(self, metrics_meters, y, y_pred_clf_1, titul, titul_pred):
        for metric_fn in self.metrics:
            if metric_fn.__name__ == 'Accuracy':
                metric_value = metric_fn(y_pred_clf_1, y).cpu().detach().numpy()
                metrics_meters['acc_document'].add(metric_value)
                metric_value = metric_fn(titul_pred, titul).cpu().detach().numpy()
                metrics_meters['acc_titul'].add(metric_value)
            else:
                metric_value = metric_fn(y_pred_clf_1, y).cpu().detach().numpy()
                metrics_meters[metric_fn.__name__].add(metric_value)

        return metrics_meters, metric_value

    def batch_update(self, x, y_clf_1, titul):
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            prediction_clf_1, titul_pred = self.model.forward(x)
        loss = self.loss_clf(prediction_clf_1, y_clf_1) * self.cls_w_1 + self.loss_clf(titul_pred, titul) * self.cls_w_2
        return loss, prediction_clf_1, titul_pred


class TestEpochCustom(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True, cls_w_1=1, cls_w_2=2, seg_w=1,
                 use_distil=False):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='test',
            device=device,
            verbose=verbose,
        )
        self.b = torch.Tensor(list(range(0, len(BACKDOCNAME2CLASS))))
        self.cls_w_1 = cls_w_1
        self.cls_w_2 = cls_w_2
        self.seg_w = seg_w
        self.optimizer = optimizer
        # self.loss_clf = FocalLoss() # nn.BCEWithLogitsLoss()
        self.loss_clf = loss
        self.use_distil = use_distil
        self.filename = '10.pdf'
        self.y_old = None
        self.pred_clf_file = []


    def run(self, valid_loader):
        device = self.device
        self.on_epoch_start()

        dict_value = dict_metric()
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics if metric.__name__ != 'accuracy'}
        metrics_meters['acc_document'] = AverageValueMeter()

        with tqdm(valid_loader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for imgs, y_all, filename in iterator:
                imgs, y_class= imgs.to(device), y_all.to(device)
                loss, preds_clf_1 = self.batch_update(imgs, y_class)
                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                metrics_meters, metric_value = self.update_batch_metrics(metrics_meters,
                                                                         y_class, preds_clf_1)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                iter_dict = list(zip(preds_clf_1, y_class, filename))

                for idx, pred_and_clas in enumerate(iter_dict):
                    pred_clf_1, y_clas, filename = pred_and_clas
                    # update table metrics
                    if filename == self.filename:
                        self.pred_clf_file.append(pred_clf_1)
                        self.y_old = y_clas
                        continue
                    else:
                        y_class_ = self.y_old
                        if len(self.pred_clf_file) != 0 :

                            pred_clf_file = torch.stack(self.pred_clf_file,1)
                            #max_el = pred_clf_file.argmax(-1)
                            #pred_clf_file = torch.zeros_like(pred_clf_file).scatter(1, max_el.unsqueeze(-1), 1.0)

                            self.pred_clf_file = [pred_clf_1]
                            pred_clf_1 = torch.median(pred_clf_file, dim=1).values

                            self.y_old = y_clas
                            self.filename = filename




                    class_str = self.class_to_str(y_class_.cpu().detach())
                    for metric_fn in self.metrics:
                        if metric_fn.__name__ != 'focal':
                            metric_value = metric_fn(pred_clf_1.unsqueeze(0), y_class_.unsqueeze(0)).cpu().detach().numpy()
                            dict_value[class_str][metric_fn.__name__].add(metric_value)

                    if self.verbose:
                        s = self._format_logs(logs)
                        iterator.set_postfix_str(s)
        return logs, dict_value

    def on_epoch_start(self):
        self.model.eval()

    def class_to_str(self, index_list):
        name = [BACKDOCNAME2CLASS[int(self.b @ index_list)]]
        return name[0]

    def update_batch_metrics(self, metrics_meters, y, y_pred_clf_1):
        for metric_fn in self.metrics:
            #if metric_fn.__name__ == 'acc_document':
            metric_value = metric_fn(y_pred_clf_1.cpu().detach(), y.cpu().detach())
            metrics_meters[metric_fn.__name__].add(metric_value)

        return metrics_meters, metric_value

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            prediction_clf_1 = self.model.forward(x)
        y_clf_1 = y
        loss = self.loss_clf(prediction_clf_1, y_clf_1) * self.cls_w_1
        return loss, prediction_clf_1