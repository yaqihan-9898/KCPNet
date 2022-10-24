from lib.datasets.voc_eval import voc_eval
from lib.datasets import pascal_voc
from lib.config import config as cfg
import os
import pickle
import numpy as np
from matplotlib import pyplot


def get_path(image_set):
    filename = image_set+'_{:s}.txt'
    path = os.path.join(
        './output/dets/' + cfg.network,
        filename)
    return path


def do_python_eval(image_set, output_dir='output'):
    devkit_path = './data/VOCdevkit2007'
    year = '2007'
    annopath = './data/VOCdevkit2007/VOC2007' + '/Annotations/' + '{:s}.xml'
    imagesetfile = os.path.join(
        devkit_path,
        'VOC2007',
        'ImageSets',
        'Main',
        image_set + '.txt')
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    p = []
    r = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(pascal_voc.CLASSES):
        if cls == '__background__':
            continue
        filename = get_path(image_set).format(cls)
        rec, prec, ap = voc_eval(
            filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)

        F = np.zeros(len(rec))
        for k in range(len(rec)):
            F[k] = 2 * prec[k] * rec[k] / (prec[k] + rec[k])
        pr_max = np.argmax(F)

        r.append(rec[pr_max])
        p.append(prec[pr_max])
        # f.append(F[pr_max])
        aps += [ap]
        print(('voc07 AP for {} = {:.4f}'.format(cls, ap)))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

    print(('Mean VOC07 AP = {:.4f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print(('{:.3f}'.format(ap)))
    print(('{:.3f}'.format(np.mean(aps))))
    print('~~~~~~~~')


if __name__ == '__main__':
    do_python_eval('test')
