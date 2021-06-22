import os
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np
import time
from nilm_models.fed_utils import *

import warnings
warnings.simplefilter("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYDEVD_USE_FRAME_EVAL'] = 'NO'

np.random.seed(0)

# import nest_asyncio
# nest_asyncio.apply()

#check_point_step_size = 100
nilmtk_sample_rate = 3
users_split = 1000
sequence_length = 101


if not Path("./dp_processed_eval_data.pkl").exists():
    from nilm_models.dae.dp_fed_dae_disaggregator import DPFedDAEDisaggregator as DPFedDisag
    import glob
    from tensorflow.keras.backend import clear_session
    from nilm_models.dae import metrics

    # Appliance Target
    utility_appliance = 'fridge'
    # NN Model input size

    h5_data_souce_path = '..\\data\\'
    users_data = Path(r'./processed_input_data.pkl').resolve()

    all_db_houses = process_nilmtk_h5(h5_data_souce_path, users_data, utility_appliance)

    test_split = 0.7
    all_dt = merge_all_houses(all_db_houses,
                              sequence_length,
                              test_split)

    tr_mains_np = all_dt[0]
    tr_appliances_gt = all_dt[1]
    test_mains_np = all_dt[2]
    test_appliances_gt = all_dt[3]
    dt_info = all_dt[4]

    all_dt_gens = preprocess_eval_generators(all_dt, 256, 5, users_split, sequence_length)
    ele_epec = all_dt_gens[4]

    attack_results = []
    model_eval_metrics = []
    model_eval_metrics2 = []

    fed_check_points = glob.glob('ckpt_*')
    fed_check_points.sort(key=os.path.getmtime)
    for fed_ckpt_path in fed_check_points:
        start = time.time()

        ckp_index = int(fed_ckpt_path.split("_")[1])
        if ckp_index > len(fed_check_points):
            break

        # Instantiating DP FED experiment class
        dp_fed_dae = DPFedDisag(sequence_length)
        dp_fed_dae.mmax = dt_info[0]

        print("--> Loading model from checkpoint {}".format(ckp_index))

        dp_fed_dae.import_model_from_ckpt(ckp_index, ele_epec, noise=0.3, server_lea=1)
        t2 = time.time()
        test_results = get_model_predictions_in_mem(dp_fed_dae,
                                                    test_mains_np,
                                                    test_appliances_gt,
                                                    dt_info)

        train_results = get_model_predictions_in_mem(dp_fed_dae,
                                                     tr_mains_np,
                                                     tr_appliances_gt,
                                                     dt_info)

        print("Time to get predictions {}".format(time.time()-t2))
        rpaf = metrics.recall_precision_accuracy_f1_v2(test_results[0], test_results[1], 50)

        rel_error = metrics.relative_error_total_energy(test_results[0], test_results[1],)
        ma_error = metrics.mean_absolute_error(test_results[0], test_results[1],)

        model_eval_metrics.append(rpaf)
        model_eval_metrics2.append((rel_error, ma_error))

        print("--> Evaluating model from checkpoint {}".format(ckp_index))

        att_results = dp_fed_dae.evaluate_model(test_results,
                                                train_results,
                                                ckp_index)

        attack_results.append(att_results)

        clear_session()

        end = time.time()
        print('Evaluation time of {}'.format(end-start))

    all_results = parse_results(attack_results, model_eval_metrics, model_eval_metrics2)

    today = date.today()

    # dd/mm/YY
    date_str = today.strftime("%b-%d-%Y")

    original_df = pd.DataFrame(all_results, dtype=np.float)
    original_df.to_pickle("./dp_processed_eval_data{}.pkl".format(date_str))
    original_df.to_pickle("./dp_processed_eval_data.pkl")
    original_df.to_excel("./dp_processed_eval_data{}.xlsx".format(date_str))
    original_df.to_excel("./dp_processed_eval_data.xlsx")

    plot_model_evaluation(all_results)

else:

    all_results_df = pd.read_pickle("./dp_processed_eval_data.pkl")
    all_results_df.to_excel("./dp_processed_eval_data.xlsx")
    all_results = all_results_df.to_dict('list')
    plot_model_evaluation(all_results)


