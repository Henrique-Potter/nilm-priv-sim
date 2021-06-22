import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


all_results_df = pd.read_pickle("./dp_processed_eval_data.pkl")
all_dp_results = all_results_df.to_dict('list')

recall_list = all_dp_results['recal']
precision_list = all_dp_results['prec']
accuracy_list = all_dp_results['acc']
f1_list = all_dp_results['f1']
rel_error = all_dp_results['rel_err']
ma_error = all_dp_results['ma_err']
ep_spent = all_dp_results['ep_spent']
membership_score = all_dp_results['membership_score']

last_recall = recall_list[-1]
last_recall = recall_list[-1]
last_recall = recall_list[-1]
last_recall = recall_list[-1]
