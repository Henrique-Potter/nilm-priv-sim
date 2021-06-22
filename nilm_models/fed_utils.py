
def create_house_list(redd_dt, utility_appliance):

    total_buildings = len(redd_dt.buildings)
    users_train_list = []

    for i in range(1, total_buildings+1):

        building_elec = redd_dt.buildings[i].elec
        building_mains = building_elec.mains().all_meters()[0]  # The aggregated meter that provides the input
        try:
            # The appliance meter target to be used as a training target
            appliance_meter = building_elec.submeters()[utility_appliance]
            # Used later for metadata structure
            users_train_list.append((building_mains, appliance_meter))

        except KeyError:
            continue

    return users_train_list


def preprocess_nilmtk_to_df(dt_source, redd_dt, utility_appliance):

    total_buildings = len(redd_dt.buildings)
    house_data_pair = {}

    for i in range(1, total_buildings+1):

        building_elec = redd_dt.buildings[i].elec
        building_mains = building_elec.mains()

        try:
            appliance_meter = building_elec.submeters()[utility_appliance]

            # Pre-processing dataset
            all_main_dt, appli_chunk = process_main_and_appliance(appliance_meter, building_mains)

            if all_main_dt is not None and appli_chunk is not None:
                print("==> Dataset {} building id {} is correlated.".format(dt_source, i))
                dt_id = "{} house {}".format(dt_source, i)
                house_data_pair[dt_id] = (all_main_dt, appli_chunk)
            else:
                print("==> Dataset {} building id {} data is not correlated.".format(dt_source, i))

        except KeyError:
            continue

    return house_data_pair


def process_main_and_appliance(appliance_meter, building_mains):
    import matplotlib.pyplot as plt

    dataset_name = appliance_meter.identifier.dataset
    building_id = appliance_meter.identifier.building

    print("\n===> Processing building id {}.".format(building_id))
    building_mains.plot()
    plt.show()
    appliance_meter.plot()
    plt.show()

    all_main_power = building_mains.power_series_all_data()
    print('==> Normal main size is: {}'.format(len(all_main_power)))
    all_appli_dt = appliance_meter.power_series_all_data()
    print('==> Normal appliance size is: {}'.format(len(all_appli_dt)))

    # all_main_power = all_main_power[all_main_power >= 1]
    # all_appli_dt = all_appli_dt[all_appli_dt >= 1]

    # Remove NaNs instead of filling them with 0s - Non existent data is not 0
    all_main_power.dropna(inplace=True)
    all_appli_dt.dropna(inplace=True)
    print('==> Main size after dropna is: {}'.format(len(all_main_power)))
    print('==> Appliance size after dropna  is: {}'.format(len(all_appli_dt)))

    ix = all_main_power.index.intersection(all_appli_dt.index)
    all_main_power = all_main_power[ix]
    all_appli_dt = all_appli_dt[ix]

    print('==> Main size after intersection is: {}'.format(len(all_main_power)))
    print('==> Appliance size after intersection is: {}'.format(len(all_appli_dt)))

    data_points_size = max(all_main_power.size, all_appli_dt.size)

    # UKdale intersection creates datasets of different sizes somehow*
    diff = abs(all_main_power.size-all_appli_dt.size)
    all_main_power = all_main_power[:data_points_size-diff]
    all_appli_dt = all_appli_dt[:data_points_size-diff]

    # all_main_power = all_main_power[:data_points_size-diff]
    # all_appli_dt = all_appli_dt[:data_points_size-diff]

    #fast_plot(all_main_power, all_appli_dt)

    print("==> Dataset size after intersection and cleaning is {}".format(len(all_main_power)))
    print("==> Appliance on meter is: {}".format(appliance_meter.on_power_threshold()))

    # if len(all_main_power) < 315000:
    #     print("--Building ignored--")
    #     return None, None

    original_corr, best_correlation, index_shift = find_max_correlation(dataset_name,
                                                                        building_id,
                                                                        all_main_power,
                                                                        all_appli_dt)
    if abs(original_corr) < 0.1:
        return None, None

    return all_main_power, all_appli_dt


plot_output_once = True


def find_max_correlation(dataset_name, building_id, series1, series2):

    import scipy.stats

    noisy_data_thres = 0.01

    new_size = len(series1)

    original_corr = scipy.stats.pearsonr(series1, series2)[0]
    best_correlation = original_corr
    index_shift = 0

    for i in range(1, 300):
        last_idx = new_size-i

        cor_dt_pos_shift = scipy.stats.pearsonr(series1.shift(periods=i)[i:], series2[i:])
        cor_dt_neg_shift = scipy.stats.pearsonr(series1.shift(periods=-i)[:last_idx], series2[:last_idx])

        if cor_dt_pos_shift[1] > noisy_data_thres and cor_dt_neg_shift[1] > noisy_data_thres:
            continue

        if cor_dt_pos_shift[0] > best_correlation:
            best_correlation = cor_dt_pos_shift[0]
            index_shift = i

        if cor_dt_neg_shift[0] > best_correlation:
            best_correlation = cor_dt_neg_shift[0]
            index_shift = -i

    if index_shift != 0:
        print("==> Better correlation found for {} building id {}.".format(dataset_name, building_id))
        print("==> Correlation:{}=>{} at index shift: {}.".format(original_corr,
                                                                  best_correlation,
                                                                  index_shift))
    else:
        print("==> Correlation:{}".format(original_corr))

    return original_corr, best_correlation, index_shift


# def get_model_predictions(tff_dp_dae, test_mains, train_appliance_meter, test_elec, model_round, plot=False):
#     from nilmtk import DataSet
#     from nilmtk.datastore import HDFDataStore
#
#     # The filename of the disagreagated prediction generated by the model
#     disag_filename = 'tff_dp_disag-out_model_{}.h5'.format(model_round)
#
#     # if not Path(disag_filename).exists():
#
#     output = HDFDataStore(disag_filename, 'w')
#
#
#     # test_mains: The aggregated signal meter
#     # output: The output datastore
#     # train_meter: This is used in order to copy the metadata of the train meter into the datastore
#     tff_dp_dae.disaggregate(test_mains, output, train_appliance_meter, sample_period=1)
#
#     output.close()
#
#     result = DataSet(disag_filename)
#     res_elec = result.buildings[1].elec
#     predicted = res_elec['fridge']
#     ground_truth = test_elec['fridge']
#
#     global plot_output_once
#     if plot:
#         import matplotlib.pyplot as plt
#         if plot_output_once:
#             ground_truth.plot()
#             plt.show()
#             plot_output_once = False
#         predicted.plot(extra_id=model_round)
#         plt.show()
#         #pred_df.plot()
#         #plt.show()
#
#     # import os
#     # os.remove(disag_filename)
#     # import tables
#     # tables.file._open_files.close_all()
#
#     return predicted, ground_truth


def fast_plot(main_dt, prediction_dt, appli_name, power_threshold_cut, loss_per_round, all_results, ground_thruth_dt=None, round_num=-1):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    main_dt[main_dt > power_threshold_cut] = power_threshold_cut
    prediction_dt[prediction_dt > power_threshold_cut] = power_threshold_cut
    ground_thruth_dt[ground_thruth_dt > power_threshold_cut] = power_threshold_cut

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2)

    # plotting predictions sample
    prediction_ax = plt.subplot(gs[0, 0])
    prediction_ax.title.set_text('Prediction')

    m_pd = pd.DataFrame(main_dt, columns=['Main meter'])
    ax1 = m_pd.plot(ax=prediction_ax, title='Prediction performance report (Round {})'.format(round_num), logy=True)

    ax1.set_xlabel('Meter samples (Mixed houses)')  # replace with the labels you want
    ax1.set_ylabel('Power (W) - Log scaled')

    ax1.plot(np.arange(len(prediction_dt)), prediction_dt, color='green', label='Prediction {} Meter'.format(appli_name))
    ax1.plot(np.arange(len(prediction_dt)), ground_thruth_dt, color='red', label='Ground truth {} Meter'.format(appli_name))

    m_pd = pd.DataFrame(ground_thruth_dt)
    m_pd.plot(ax=prediction_ax)

    ax1.legend(bbox_to_anchor=(1.105, 1), shadow=True, fancybox=True)
    ax1.legend(['Main meter', "Prediction {} Meter".format(appli_name), "Ground truth {} Meter".format(appli_name)])

    # plotting loss subplot
    loss_ax = plt.subplot(gs[1, :])
    loss_ax.plot(np.arange(len(loss_per_round)),
                 loss_per_round,
                 color='blue',
                 linewidth=2,
                 linestyle='solid',
                 label="Training  Loss")

    loss_ax.legend(bbox_to_anchor=(1.105, 1), shadow=True, fancybox=True)

    # plotting eval metrics sample
    eval_metrics_ax = plt.subplot(gs[0, 1])

    df = pd.DataFrame({'Metrics': ['recal', 'prec', 'acc', 'f1', 'rel_err', 'ma_err'], 'values': all_results})
    ax = df.plot.bar(ax=eval_metrics_ax, x='Metrics', y='values', logy=True)

    ax.set_ylabel('Scores')
    ax.set_title('Model metrics for round {}'.format(round_num))

    for p in ax.patches:
        ax.annotate(str(round(p.get_height(), 2)), (p.get_x() * 1.005, p.get_height() * 1.01))

    plt.savefig('plots/model_conver_rd_{}_{}.png'.format(round_num, appli_name))
    plt.show()


def round_data_to_model_input(main_data, appliance_data, sequence_length):

    dt_size = len(main_data)
    # Clipping data to fit model input
    s = sequence_length
    additional = dt_size % s
    main_data = main_data[:dt_size - additional]
    appliance_data = appliance_data[:dt_size - additional]

    return main_data, appliance_data


def merge_all_houses(all_db_houses, model_input_len, test_split_rate, model_preprocess):
    import numpy as np

    tr_mains_np = np.empty(0, dtype=float)
    tr_appli_gt = np.empty(0, dtype=float)

    tt_mains_np = np.empty(0, dtype=float)
    tt_appli_gt = np.empty(0, dtype=float)

    print(" --- Merging {} houses datasets --- ".format(len(all_db_houses)))

    for key, value in all_db_houses.items():
        dt_len = len(value[0])
        index_split = int(dt_len * test_split_rate)

        tr_mains_np = np.append(tr_mains_np, value[0][:index_split])
        tr_appli_gt = np.append(tr_appli_gt, value[1][:index_split])

        tt_mains_np = np.append(tt_mains_np, value[0][index_split:dt_len])
        tt_appli_gt = np.append(tt_appli_gt, value[1][index_split:dt_len])

    tr_mains_np, tr_appli_gt = round_data_to_model_input(tr_mains_np, tr_appli_gt, model_input_len)

    tt_mains_np, tt_appli_gt = round_data_to_model_input(tt_mains_np, tt_appli_gt, model_input_len)

    print(" ==> Train Main Meter dataset size: {} ".format(len(tr_mains_np)))
    print(" ==> Train Appliance Meter dataset size: {} ".format(len(tr_appli_gt)))
    print(" ==> Test Main dataset size: {} ".format(len(tt_mains_np)))
    print(" ==> Test Appliance dataset size: {} ".format(len(tt_appli_gt)))

    tr_mains_np, tr_appli_gt, tt_mains_np, tt_appli_gt = model_preprocess(tr_mains_np, tr_appli_gt, tt_mains_np, tt_appli_gt)

    return tr_mains_np, tr_appli_gt, tt_mains_np, tt_appli_gt


def normalize(chunk, mmax):
    '''Normalizes timeseries

    Parameters
    ----------
    chunk : the timeseries to normalize
    max : max value of the powerseries

    Returns: Normalized timeseries
    '''
    tchunk = chunk / mmax
    return tchunk


def denormalize(chunk, mmax):
    '''Deormalizes timeseries
    Note: This is not entirely correct

    Parameters
    ----------
    chunk : the timeseries to denormalize
    max : max value used for normalization

    Returns: Denormalized timeseries
    '''
    tchunk = chunk * mmax
    return tchunk


def get_model_predictions_in_mem(disag_model,
                                 main_dt,
                                 appliance_dt,
                                 plot=False):
    import numpy as np
    import pandas as pd

    pred_df, ground_truth_df, main_dt, point_wise_loss = disag_model.disaggregate_in_mem(main_dt, appliance_dt)

    main_column = pd.Series(main_dt, name='Main meters')
    main_powers_dict = {'Main meters': main_column}
    main_powers = pd.DataFrame(main_powers_dict)

    idxs = np.arange(pred_df.size)

    global plot_output_once
    if plot:
        import matplotlib.pyplot as plt
        if plot_output_once:
            main_ax = main_powers.plot()

            main_ax.set_xlabel('Time in Seconds')  # replace with the labels you want
            main_ax.set_ylabel('Main total Power (W)')
            plt.show()

            ground_ax = ground_truth_df.plot()

            ground_on_markers = np.array([p if p > 50 else None for p in ground_truth_df.iloc[:, 0]])
            ground_off_markers = np.array([p if p < 50 else None for p in ground_truth_df.iloc[:, 0]])

            ground_ax.plot(idxs, ground_on_markers, color='green', linewidth=1, linestyle='solid', label='ON')
            ground_ax.plot(idxs, ground_off_markers, color='red', linewidth=1, linestyle='solid', label='OFF')
            ground_ax.set_xlabel('Time in Seconds (Matches the input time-wise)')  # replace with the labels you want
            ground_ax.set_ylabel('Ground Power (W)')
            plt.show()

            plot_output_once = False

            pred_ax = pred_df.plot()
            pred_ax_on_markers = np.array([p if p > 50 else None for p in pred_df.iloc[:, 0]])
            pred_ax_off_markers = np.array([p if p < 50 else None for p in pred_df.iloc[:, 0]])

            pred_ax.plot(idxs, pred_ax_on_markers, color='green', linewidth=1, linestyle='solid', label='ON')
            pred_ax.plot(idxs, pred_ax_off_markers, color='red', linewidth=1, linestyle='solid', label='OFF')
            pred_ax.set_xlabel('Time in Seconds (Matches the input time-wise)')  # replace with the labels you want
            pred_ax.set_ylabel('Predicted Power (W)')
            plt.show()

    return pred_df, ground_truth_df, point_wise_loss


def preprocess(sequence_length, dataset, epochs, batch_size, prefetch=10000):
    import tensorflow as tf
    import collections

    @tf.function
    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            x=tf.reshape(element['x'], [-1, sequence_length, 1]),
            y=tf.reshape(element['y'], [-1, 1, 1]))

    # x = tf.reshape(element['x'], [-1, sequence_length, 1]),
    # y = tf.reshape(element['y'], [-1, 1, 1]))

    dataset = dataset.repeat(epochs).batch(batch_size, drop_remainder=True).map(batch_format_fn, num_parallel_calls=3).prefetch(512)

    return dataset


def preprocess_eval_generators(all_dt, generator_pp, batch_size, epochs, synth_users_split, sequence_len):

    tr_main_dt_len = int(len(all_dt[0]))
    tt_main_dt_len = int(len(all_dt[2]))

    train_mains_np = all_dt[0][:tr_main_dt_len]
    train_appliances_gt = all_dt[1][:tr_main_dt_len]
    test_mains_np = all_dt[2][:tt_main_dt_len]
    test_appliances_gt = all_dt[3][:tt_main_dt_len]

    print(" Training dataset size is {}".format(len(train_mains_np)))
    print(" Test dataset size is {}".format(len(test_mains_np)))

    train_dt_gens, train_labels, element_spec = create_model_dt_generators(train_mains_np,
                                                                           train_appliances_gt,
                                                                           batch_size,
                                                                           epochs,
                                                                           synth_users_split,
                                                                           sequence_len,
                                                                           generator_pp)

    test_dt_gens, test_labels, _ = create_model_dt_generators(test_mains_np,
                                                              test_appliances_gt,
                                                              batch_size,
                                                              epochs,
                                                              synth_users_split,
                                                              sequence_len,
                                                              generator_pp)

    return train_dt_gens, train_labels, test_dt_gens, test_labels, element_spec


def create_model_dt_generators(mains_dt,
                               appli_dt,
                               batch_size,
                               epochs,
                               synth_users_split,
                               sequence_length,
                               generator_pp):
    import numpy as np
    import tensorflow as tf

    data_points_size = len(mains_dt)
    # Clipping data to fit model input
    s = sequence_length

    additional = data_points_size % s
    x_data = mains_dt[:data_points_size - additional]
    y_data = appli_dt[:data_points_size - additional]

    x_data, y_data = generator_pp(sequence_length, x_data, y_data)

    # x_batch = np.reshape(x_data, (int(len(x_data) / s), s, 1))
    # y_batch = np.reshape(y_data, (int(len(y_data) / s), s, 1))

    print(" -> Number of data points from all houses: {}".format(x_data.shape[0]))
    print(" -> Average user dataset size: {}".format(int(x_data.shape[0] / synth_users_split)))
    split_x_batch = np.array_split(x_data, synth_users_split)
    split_y_batch = np.array_split(y_data, synth_users_split)

    users_processed_data = []
    users_labels_np = []
    element_spec = None

    for x_split, y_split in zip(split_x_batch, split_y_batch):
        temp_tensor_dt = tf.data.Dataset.from_tensor_slices({"x": x_split, "y": y_split})
        #temp_tensor_dt = tf.data.Dataset.from_tensor_slices((x_split, y_split))
        split_dt_prefetcher = preprocess(sequence_length, temp_tensor_dt, epochs, batch_size)
        element_spec = split_dt_prefetcher.element_spec
        users_processed_data.append(split_dt_prefetcher)
        users_labels_np.append(y_split)

    return users_processed_data, users_labels_np, element_spec


def process_nilmtk_h5(data_source: str, users_data, utility_appliance):

    from nilmtk import DataSet
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import time
    import glob

    data_source = glob.glob(data_source + '*.h5')

    all_db_houses = {}

    if not users_data.exists():
        print("Loading Datasets")
        for data_source_path in data_source:

            if data_source_path in "..\\data\\SynD.h5" or data_source_path in "..\\data\\iawe.h5":
                print('Skipping {}'.format(data_source_path))
                continue

            dt = DataSet(data_source_path)
            file_name = Path(data_source_path).name
            print("\n\n===>Pre-processing {} dataset".format(file_name))

            start = time.time()
            house_data_pair = preprocess_nilmtk_to_df(file_name, dt, utility_appliance)
            print("=>Time to process {} data: {}s ".format(data_source_path, time.time() - start))

            all_db_houses.update(house_data_pair)

        df = pd.DataFrame(all_db_houses, dtype=np.float)
        df.to_hdf("./processed_input_data.pkl", key='df', mode='w')

    else:
        df = pd.read_hdf("./processed_input_data.pkl", key='df')
        all_db_houses = df.to_dict('series')
    return all_db_houses


def last_4chars(x):
    return(x[-4:])


def get_max_value(redd_dt, utility_appliance, **load_kwargs):

    house_list = create_house_list(redd_dt, utility_appliance)

    n_users = len(house_list)

    highest_power = 0

    for i in range(n_users):

        if i is 1:
            continue

        main_power_series = house_list[i][0].power_series(**load_kwargs)
        meter_power_series = house_list[i][1].power_series(**load_kwargs)
        run = True
        main_chunk = next(main_power_series)

        while run:
            if highest_power < main_chunk.max():
                highest_power = main_chunk.max()
            try:
                main_chunk = next(main_power_series)
                meter_chunk = next(meter_power_series)
            except:
                run = False

    return highest_power


def plot_model_evaluation(all_results):

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    recall_list = all_results['recal']
    precision_list = all_results['prec']
    accuracy_list = all_results['acc']
    f1_list = all_results['f1']
    rel_error = all_results['rel_err']
    ma_error = all_results['ma_err']
    ep_spent = all_results['ep_spent']

    try:
        membership_score = all_results['membership_score']
    except:
        membership_score = []

    total_train_loss = all_results['total_tr_loss']
    total_test_loss = all_results['total_tt_loss']

    nr_reports = len(recall_list)
    round_number = np.arange(1, nr_reports, step=1)
    round_number_list = ['Round {}'.format(i) for i in range(nr_reports)]

    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(2, 3)

    model_metrics_1_ax = plt.subplot(gs[0, 0])
    model_metrics_1_ax.title.set_text('Model Metrics 1')

    model_metrics_2_ax = plt.subplot(gs[0, 1])
    model_metrics_2_ax.title.set_text('Model Metrics 2')

    attack_metrics_ax = plt.subplot(gs[0, 2])
    attack_metrics_ax.title.set_text('Attack Metrics')

    loss_ax = plt.subplot(gs[1, :])
    loss_ax.title.set_text('Loss')

    model_metrics_1_ax.plot(round_number, recall_list, color='red', linewidth=2, linestyle='solid', label='Recall')
    model_metrics_1_ax.plot(round_number, precision_list, color='blue', linewidth=2, linestyle='dashed', label='Precision')
    model_metrics_1_ax.plot(round_number, accuracy_list, color='green', linewidth=2, linestyle='dotted', label='Accuracy')
    model_metrics_1_ax.plot(round_number, f1_list, color='yellow', linewidth=2, linestyle='dashdot', label='F1')
    model_metrics_1_ax.plot(round_number, rel_error, color='black', linewidth=2, linestyle='solid', label='RE')
    model_metrics_1_ax.legend(bbox_to_anchor=(1.04, 1), shadow=True, fancybox=True)
    #model_metrics_1_ax.set_xticks(round_number, minor=False)
    #model_metrics_1_ax.set_xticklabels(round_number_list, fontdict=None, minor=False)

    model_metrics_2_ax.plot(round_number, ma_error, color='blue', linewidth=2, linestyle='dashed', label='MAE')
    model_metrics_2_ax.legend(bbox_to_anchor=(1.04, 1), shadow=True, fancybox=True)
    #model_metrics_2_ax.set_xticks(round_number, minor=False)
    #model_metrics_2_ax.set_xticklabels(round_number_list, fontdict=None, minor=False)

    if len(membership_score) is not 0:
        attack_metrics_ax.plot(round_number, membership_score, color='blue', linewidth=2, linestyle='solid', label="Membership Score")

        attack_metrics_ax.legend(bbox_to_anchor=(1.04, 1), shadow=True, fancybox=True)
        #attack_metrics_ax.set_xticks(round_number, minor=False)
        #attack_metrics_ax.set_xticklabels(round_number_list, fontdict=None, minor=False)

    loss_ax.plot(round_number, total_train_loss, color='blue', linewidth=2, linestyle='solid', label="Tr L")
    loss_ax.plot(round_number, total_test_loss, color='green', linewidth=2, linestyle='dashed', label="Te L")

    loss_ax.legend(bbox_to_anchor=(1.105, 1), shadow=True, fancybox=True)
    #loss_ax.set_xticks(round_number, minor=False)
    #loss_ax.set_xticklabels(round_number_list, fontdict=None, minor=False)

    plt.subplots_adjust(left=0.07, wspace=1.5, hspace=0.35)
    plt.savefig('fed_model_SGD_learning_1.png')
    plt.show()


def parse_results(attack_reports, model_eval_metrics, model_eval_metrics2):

    all_results = {}

    recal_list = []
    precision_list = []
    accuracy_list = []
    f1_list = []
    rel_error = []
    ma_error = []
    membership_score = []

    total_train_loss = []
    total_test_loss = []
    ep_spent = []

    for at_rep, mt, mt2 in zip(attack_reports, model_eval_metrics, model_eval_metrics2):
        recal_list.append(mt[0])
        precision_list.append(mt[1])
        accuracy_list.append(mt[2])
        f1_list.append(mt[3])

        rel_error.append(mt2[0])
        ma_error.append(mt2[1])

        if at_rep[0] is not None:
            membership_score.append(at_rep[0])

        total_train_loss.append(at_rep[1])
        total_test_loss.append(at_rep[2])

        ep_spent.append(at_rep[3])

    all_results['recal'] = recal_list
    all_results['prec'] = precision_list
    all_results['acc'] = accuracy_list
    all_results['f1'] = f1_list
    all_results['rel_err'] = rel_error
    all_results['ma_err'] = ma_error

    if at_rep[0] is not None:
        all_results['membership_score'] = membership_score

    all_results['total_tr_loss'] = total_train_loss
    all_results['total_tt_loss'] = total_test_loss
    all_results['ep_spent'] = ep_spent

    return all_results


def fast_metrics_bar_plot(all_results, round_number):
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.figure(figsize=(20, 3))
    df = pd.DataFrame({'Metrics': ['recal', 'prec', 'acc', 'f1', 'rel_err', 'ma_err'], 'values': all_results})

    ax = df.plot.bar(x='Metrics', y='values', logy=True)

    ax.set_ylabel('Scores')
    ax.set_title('Model metrics for round {}'.format(round_number))

    for p in ax.patches:
        ax.annotate(str(round(p.get_height(), 2)), (p.get_x() * 1.005, p.get_height() * 1.01))

    plt.show()


def compute_epsilon(steps, noise_multi, user_ratio, delta):

    from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
    from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
    """Computes epsilon value for given hyperparameters."""
    if noise_multi == 0.0:
        return float('inf')

    # This probably are the alphas
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))

    rdp = compute_rdp(
      q=user_ratio,
      noise_multiplier=noise_multi,
      steps=steps,
      orders=orders)
    # Delta is set to 1e-5 because MNIST has 60000 training points.
    return get_privacy_spent(orders, rdp, target_delta=delta)[0]
