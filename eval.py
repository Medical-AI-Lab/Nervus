#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
from pathlib import Path
import re
import pandas as pd

from metrics.roc import cal_roc
# from metrics.yy import cal_yy
# from metrics.c_index import cal_c_index

from logger.logger import Logger

logger = Logger.get_logger('eval')


parser = argparse.ArgumentParser(description='Options for eval')
parser.add_argument('--eval_datetime', type=str, default=None, help='date time for evaluation(Default: None)')
args = parser.parse_args()


def _get_latest_test_datetime():
    date_names = [path for path in Path('./results/sets/').glob('*') if re.search(r'\d+', str(path))]
    latest = max(date_names, key=lambda date_name: date_name.stat().st_mtime).name
    return latest


def define_cal_metrics(task):
    if task == 'classification':
        return cal_roc
    elif task == 'regression':
        pass
        # return cal_yy
    elif task == 'deepsurv':
        pass
        # return cal_c_index
    else:
        logger.error(f"Invalid task: {task}.")


def save_fig():
    pass


def update_summary():
    pass


# Check date time
if args.eval_datetime is None:
    args.eval_datetime = _get_latest_test_datetime()


# Check task
parameter_path = Path('./results/sets', args.eval_datetime, 'parameter.csv')
df_args = pd.read_csv(parameter_path)
task = df_args.loc[df_args['option'] == 'task', 'parameter'].item()
likelihood_paths = Path('./results/sets/', args.eval_datetime, 'likelihoods').glob('likelihood_*.csv')
cal_metrics = define_cal_metrics(task)


logger.info(f"Calucating metrics for {args.eval_datetime}.")
for likelihood_path in likelihood_paths:
    logger.info(f"Load {likelihood_path.stem}.")

    metrics, fig = cal_metrics(likelihood_path)

    # if fig is not None:
    # Save fig

    # Update summary

logger.info(f"Calculated metrics for {args.eval_datetime}.")


"""
# Save ROC
        roc_name = institution + '_' + likelihood_name.replace(nervusenv.csv_name_likelihood, nervusenv.roc_name).replace('.csv', '.png')   # roc_weight_epoch-004-best.png
        roc_dir = os.path.join(datetime_dir, nervusenv.roc_dir)
        os.makedirs(roc_dir, exist_ok=True)
        roc_path = os.path.join(roc_dir, roc_name)
        plt.savefig(roc_path)


        # Save AUC
        datetime = os.path.basename(datetime_dir)
        weight_name = likelihood_name.replace(nervusenv.csv_name_likelihood + '_', '').replace('.csv', '.pt')   # weight_epoch-004-best.pt
        summary_tmp = dict()
        summary_tmp['datetime'] = [datetime]
        summary_tmp['weight'] = [weight_name]
        summary_tmp['Institution'] = [institution]
        for label_name, label_roc in metrics_roc.items():
            auc_val = label_roc.val.auc.micro if (label_roc.val.auc.micro is not None) else label_roc.val.auc.macro
            auc_test = label_roc.test.auc.micro if (label_roc.test.auc.micro is not None) else label_roc.test.auc.macro
            summary_tmp[label_name+'_val_auc'] = [f"{auc_val:.2f}"]
            summary_tmp[label_name+'_test_auc'] = [f"{auc_test:.2f}"]

        df_summary_tmp = pd.DataFrame(summary_tmp)
        df_summary_new = pd.concat([df_summary_new, df_summary_tmp], ignore_index=True)


# Update summary
#df_summary_new = pd.DataFrame(summary_new)
update_summary(nervusenv.summary_dir, nervusenv.csv_summary, df_summary_new)
"""