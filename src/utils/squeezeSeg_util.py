""" Utility Functions """

import numpy as np
import os

import torch
from torch.nn.parameter import Parameter

CLS_COLOR_MAP = np.array([[0.00, 0.00, 0.00],
                          [0.12, 0.56, 0.37],
                          [0.66, 0.55, 0.71],
                          [0.58, 0.72, 0.88]])

def img_normalize(x):
    return ((x - torch.min(x)) / (torch.max(x) - torch.min(x))).view(
        x.size()[0], x.size()[1]).cpu().numpy() * 255

def visualize_seg(label_map, one_hot=False):
    if one_hot:
        label_map = np.argmax(label_map, axis=-1)

    out = torch.zeros([label_map.size()[0], label_map.size()[1], 3], dtype=torch.float64)

    for l in range(1, 4):
        out[label_map==l, :] = torch.from_numpy(CLS_COLOR_MAP[l])

    return out.cpu().numpy() * 255

def bgr_to_rgb(ims):
    """Convert a list of images from BGR format to RGB format."""
    out = []
    for im in ims:
        out.append(im[:,:,::-1])

    return out

def evaluate(label, pred, n_class):
    """Evaluation script to compute pixel level IoU.

    Args:
        label: N-d array of shape [batch, W, H], where each element is a class index.
        pred: N-d array of shape [batch, W, H], the each element is the predicted class index.
        n_class: number of classes
        epsilon: a small value to prevent division by 0

    Returns:
        IoU: array of lengh n_class, where each element is the average IoU for this class.
        tps: same shape as IoU, where each element is the number of TP for each class.
        fps: same shape as IoU, where each element is the number of FP for each class.
        fns: same shape as IoU, where each element is the number of FN for each class.
    """

    assert label.shape == pred.shape, \
        'label and pred shape mismatch: {} vs {}'.format(label.shape, pred.shape)

    label = label.cpu().numpy()
    pred = pred.cpu().numpy()

    tp = np.zeros(n_class)
    fn = np.zeros(n_class)
    fp = np.zeros(n_class)

    for cls_id in range(n_class):
        tp_cls = np.sum(pred[label == cls_id] == cls_id)
        fp_cls = np.sum(label[pred == cls_id] != cls_id)
        fn_cls = np.sum(pred[label == cls_id] != cls_id)

        tp[cls_id] = tp_cls
        fp[cls_id] = fp_cls
        fn[cls_id] = fn_cls

    return tp, fp, fn

def print_evaluate(name, value, classes):
    print(f'{name}:')
    for i in range(1, len(classes)):
        print(f'{classes[i]}: {value[i]}')
    print()

def save_checkpoint(model_dir, network, epoch, model):
    save_path = os.path.join(model_dir, f"{network.lower()}_epoch_{epoch}.pth")
    torch.save(model.module.state_dict(), save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(model_dir, network, epoch, model):
    load_path = os.path.join(model_dir, f"{network.lower()}_epoch_{epoch}.pth")
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint)
    print(f"Checkpoint loaded to {load_path}")