import numpy as np

def calcCenter(x, y, field_abs):

    field_norm = field_abs / np.max(field_abs)

    M0 = np.sum(field_norm)

    xm = np.sum(x * field_norm) / M0
    ym = np.sum(y * field_norm) / M0

    return xm, ym
