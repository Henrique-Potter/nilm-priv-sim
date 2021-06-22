import numpy as np


def tp_tn_fp_fn(pred_power_series, ground_power_series, threshold):

    pr = pred_power_series > threshold
    gr = ground_power_series > threshold

    tp = float(np.sum(np.logical_and(pr, gr)))
    fp = float(np.sum(np.logical_and(pr, gr == 0)))
    fn = float(np.sum(np.logical_and(pr == 0, gr)))
    tn = float(np.sum(np.logical_and(pr == 0, gr == 0)))

    return tp, tn, fp, fn


def recall_precision_accuracy_f1_v2(pred, ground, threshold):

    tp, tn, fp, fn = tp_tn_fp_fn(pred, ground, threshold)
    total_predictions = len(pred)

    res_recall = recall(tp, fn)
    res_precision = precision(tp, fp)
    res_f1 = f1(res_precision, res_recall)
    res_accuracy = accuracy(tp, tn, total_predictions)

    return res_recall, res_precision, res_accuracy, res_f1


def get_non_na_intersection(source_of_na, target, plot=False):

    notna_booleans = source_of_na.notna()
    notna_indexes = notna_booleans[notna_booleans]
    not_na_ix = source_of_na.index.intersection(notna_indexes.index)

    non_na_source = source_of_na[not_na_ix]
    non_na_target = target[not_na_ix]

    if plot:
        import matplotlib.pyplot as plt
        source_of_na.plot()
        plt.show()
        non_na_source.plot()
        plt.show()
        target.plot()
        plt.show()
        non_na_target.plot()
        plt.show()

    return non_na_source, non_na_target


def relative_error_total_energy(pred, ground):
    import pandas as pd

    if isinstance(pred, pd.DataFrame):
        pred_summed = np.sum(pred.values)
        ground_summed = np.sum(ground.values)
    else:
        pred_summed = np.sum(pred)
        ground_summed = np.sum(ground)

    return abs(pred_summed - ground_summed) / float(max(pred_summed, ground_summed))


def mean_absolute_error(pred, ground):
    import pandas as pd

    if isinstance(pred, pd.DataFrame):
        total_sum = np.sum(abs(pred.values - ground.values))
    else:
        total_sum = np.sum(abs(pred - ground))

    return total_sum / len(pred)


def recall(tp, fn):
    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        return 0

    return recall


def precision(tp, fp):
    try:
        pres = tp / float(tp + fp)
    except ZeroDivisionError:
        return 0

    return pres


def f1(prec, rec):
    try:
        f1_mes = 2 * (prec * rec) / float(prec + rec)
    except ZeroDivisionError:
        return 0

    return f1_mes


def accuracy(tp, tn, total_predictions):
    return (tp + tn) / float(total_predictions)
