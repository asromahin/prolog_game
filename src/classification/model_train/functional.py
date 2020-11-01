import torch


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        #return (x > threshold).type(x.dtype)
        #x = torch.Tensor(x)
        max_el = x.argmax(-1)
        x = torch.zeros_like(x).scatter(1, max_el.unsqueeze(-1), 1.0)
        return x
    else:
        return x

def _activation_pr(pr, activation):
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    elif activation == "hardsigmoid":
        activation_fn = torch.nn.Hardsigmoid()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )
    pr = activation_fn(pr)
    return pr



def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None, activation=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
        ignore_channels(bull): if you want ignore
        activation (str: sigmoid, softmax2d)
    Returns:
        float: F score
    """
    pr = _activation_pr(pr, activation)
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)


    tp = torch.sum((gt * pr))
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    presion = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    score = ((1 + beta ** 2) * presion * recall + eps) / (beta ** 2 * presion + recall + eps)
    #score = ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


def accuracy(pr, gt, threshold=0.5, ignore_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)


    tp = torch.sum((gt == pr).all(dim=1))
    score = tp.float() / pr.size(0)
    return score


def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score


def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score