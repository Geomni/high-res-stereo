import os
import os.path


def dataloader(filepath):
    _, dirs, _ = list(os.walk(filepath))[0]
    left_train = ['%s/%s/im0.png' % (filepath, sample_dir) for sample_dir in dirs]
    right_train = ['%s/%s/im1.png' % (filepath, sample_dir) for sample_dir in dirs]
    disp_train_l = ['%s/%s/disp0GT.pfm' % (filepath, sample_dir) for sample_dir in dirs]
    disp_train_r = ['%s/%s/disp1GT.pfm' % (filepath, sample_dir) for sample_dir in dirs]
    return left_train, right_train, disp_train_l, disp_train_r
