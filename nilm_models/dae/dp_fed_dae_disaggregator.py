from __future__ import print_function, division
import collections
import glob
import os

import numpy as np

import matplotlib.pyplot as plt
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from pathlib import Path
import pandas as pd
import h5py
import random
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Reshape, Dropout, GRU, Bidirectional, LSTM
from tensorflow.keras.utils import plot_model

from nilm_models.StateCheckPoint import FileCheckpointManager
from nilm_models.dae import metrics
from nilm_models.fed_utils import compute_epsilon, round_data_to_model_input, denormalize, create_model_dt_generators, \
    fast_plot, fast_metrics_bar_plot

from nilmtk.legacy.disaggregate import Disaggregator

import tensorflow as tf
import tensorflow_federated as tff

#tf.keras.mixed_precision.experimental.set_policy('float64')
#tf.keras.mixed_precision.set_global_policy('float64')

global_input_spec = None
gl_mdl_in_len = 0


def client_weight_fn(local_outputs):
    del local_outputs
    return 1.0


def create_dp_optimizer(l2_clip, noise_multi, micro_batches, learning_rate):

    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=l2_clip,
        noise_multiplier=noise_multi,
        num_microbatches=micro_batches,
        learning_rate=learning_rate)

    return optimizer


def create_model_wgru(window_size):
    '''Creates the GRU architecture described in the paper
    '''
    model = Sequential()

    # 1D Conv
    model.add(Conv1D(16, 4, activation='relu', input_shape=(window_size, 1), padding="same", strides=1))

    #Bi-directional GRUs
    model.add(Bidirectional(GRU(64, activation='relu', return_sequences=True), merge_mode='concat'))
    model.add(Dropout(0.5))
    model.add(Bidirectional(GRU(128, activation='relu', return_sequences=False), merge_mode='concat'))
    model.add(Dropout(0.5))

    # Fully Connected Layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    #model.compile(loss='mse', optimizer='adam')
    #print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)

    return model


def create_model_rnn():
    '''Creates the RNN module described in the paper
    '''
    model = Sequential()

    # 1D Conv
    model.add(Conv1D(16, 4, activation="linear", input_shape=(1, 1), padding="same", strides=1))

    #Bi-directional LSTMs
    model.add(Bidirectional(LSTM(128, return_sequences=True, stateful=False), merge_mode='concat'))
    model.add(Bidirectional(LSTM(256, return_sequences=False, stateful=False), merge_mode='concat'))

    # Fully Connected Layers
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(1, activation='linear'))

    #model.compile(loss='mse', optimizer='adam')
    plot_model(model, to_file='model.png', show_shapes=True)

    return model


def create_model_s2p(sequence):
    '''Creates and returns the ShortSeq2Point Network
    Based on: https://arxiv.org/pdf/1612.09106v3.pdf
    '''
    model = Sequential()
    dropout = 0.1
    # 1D Conv
    model.add(Conv1D(30, 10, activation='relu', input_shape=(sequence, 1), padding="same", strides=1))
    model.add(Conv1D(30, 8, activation='relu', padding="same", strides=1))
    model.add(Dropout(dropout))
    model.add(Conv1D(40, 6, activation='relu', padding="same", strides=1))
    model.add(Dropout(dropout))
    model.add(Conv1D(50, 5, activation='relu', padding="same", strides=1))
    model.add(Dropout(dropout))
    model.add(Conv1D(50, 5, activation='relu', padding="same", strides=1))
    model.add(Dropout(dropout))
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))

    #model.compile(loss='mse', optimizer='adam')
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)

    return model


def create_model(sequence_len):
    '''Creates the Auto encoder module described in the paper
    '''
    model = Sequential()

    # 1D Conv
    model.add(Conv1D(8, 4, activation="linear", input_shape=(sequence_len, 1), padding="same", strides=1))
    model.add(Flatten())

    # Fully Connected Layers
    model.add(Dropout(0.2))
    model.add(Dense((sequence_len - 0) * 8, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense((sequence_len - 0) * 8, activation='relu'))

    model.add(Dropout(0.2))

    # 1D Conv
    model.add(Reshape(((sequence_len - 0), 8)))
    model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))

    # model.compile(loss='mse', optimizer='adam')
    plot_model(model, to_file='model.png', show_shapes=True)

    return model


def loss_fn_federated(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))


def element_spec_dict(element, sequence_len):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
        x=tf.reshape(element['x'], [-1, sequence_len, 1]),
        y=tf.reshape(element['y'], [-1, 1, 1]))


def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = gl_mdl_builder(gl_mdl_in_len)
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=global_input_spec,
        loss=gl_loss_func)


class DPFedDAEDisaggregator(Disaggregator):

    def __init__(self, appli_name, on_threshold, mdl_name, keras_mdl_builder, loss_func, input_len, dt_prep, dt_postp, gen_pp, in_reshape):

        self.appli_name = appli_name
        self.on_threshold = on_threshold
        self.mdl_name = mdl_name
        self.ckp_dir_name = "./{}_{}".format(self.mdl_name, self.appli_name)
        self.trained_mdl = None

        # Dealing with RuntimeError: Attempting to capture an EagerTensor without building a function.
        global gl_mdl_in_len
        gl_mdl_in_len = input_len
        self.mdl_in_len = input_len

        global gl_mdl_builder
        gl_mdl_builder = keras_mdl_builder

        global gl_loss_func
        gl_loss_func = loss_func

        self.data_preprocess = dt_prep
        self.data_postprocess = dt_postp

        self.fed_state = None
        self.eps_per_round_list = None
        self.element_spec = None
        self.mmax = None
        self.gen_pp = gen_pp
        self.in_reshape = in_reshape

    # Trains the Fed Model
    def train(self,
              all_dt,
              epochs,
              batch_size,
              synth_users_split,
              user_sample_size,
              nr_rounds,
              l2_clip,
              noise_multi,
              check_point_rate,
              server_learning_rate,
              sample_plot_rate=-1):

        train_mains_np = all_dt[0]
        train_appliances_gt = all_dt[1]
        test_main = all_dt[2]
        test_appli = all_dt[3]

        print(" Training dataset size is {}".format(len(train_mains_np)))
        print(" Test dataset size is {}".format(len(all_dt[2])))

        train_dt_gens, _, element_spec = create_model_dt_generators(train_mains_np,
                                                                    train_appliances_gt,
                                                                    batch_size,
                                                                    epochs,
                                                                    synth_users_split,
                                                                    self.mdl_in_len,
                                                                    self.gen_pp)

        global global_input_spec
        global_input_spec = element_spec

        if user_sample_size > 1:

            self.fed_train(check_point_rate, l2_clip, noise_multi, nr_rounds, sample_plot_rate, server_learning_rate,
                           test_appli, test_main, train_dt_gens, user_sample_size)
        else:
            model = gl_mdl_builder(gl_mdl_in_len)
            model.compile(loss='mse', optimizer='adam')
            model.fit(train_dt_gens[0], epochs=5)

        # Saving the loss

        # # Getting references to the last Fed state and trained model
        # keras_model = create_model(self.sequence_length)
        # keras_model.compile(loss=tf.keras.losses.MeanSquaredError())
        # state.model.assign_weights_to(keras_model)
        # self.model = keras_model
        # self.fed_state = state

    def fed_train(self, check_point_rate, l2_clip, noise_multi, nr_rounds, sample_plot_rate, server_learning_rate,
                  test_appli, test_main, train_dt_gens, user_sample_size):

        iterative_process = self.build_fed_avg(l2_clip, noise_multi, server_learning_rate)
        # First round of updates for the fed model
        state = iterative_process.initialize()
        # Load the latest state based on directory creation time
        ckpts_list = glob.glob(self.ckp_dir_name + '/ckpt_*')
        ckpts_list.sort(key=os.path.getmtime)
        first_round = 1
        if len(ckpts_list) is not 0:
            last_saved_round = int(ckpts_list[-1].split("_")[-1])
            state = self.load_from_ckpt(state, last_saved_round)
            # The next round of training
            first_round = last_saved_round + 1
        loss_per_round = []
        # Fed Training Loop
        for round_num in range(first_round, nr_rounds + 1):
            start = time.time()

            # Random sub sample of users
            selected_users = random.sample(train_dt_gens, user_sample_size)
            # 1 Round of client training and fed model update
            state, metrics = iterative_process.next(state, selected_users)
            print('=>Round {:2d}, metrics={}'.format(round_num, metrics))

            end = time.time()

            round_loss = metrics['train']['loss']
            loss_per_round.append(round_loss)

            print('--Training time of {}'.format(end - start))
            if round_num % check_point_rate is 0:
                import matplotlib.pyplot as plt

                Path(self.ckp_dir_name).mkdir(parents=True, exist_ok=True)
                # Create the checkpoint Manager
                ckpt_manager = FileCheckpointManager(root_dir=self.ckp_dir_name)
                # Save checkpoint for round round_num
                ckpt_manager.save_checkpoint(state, round_num=round_num)

            if round_num % sample_plot_rate is 0:
                self.disaggregate_test(state, test_main, test_appli, loss_per_round, round_num)

    def build_fed_avg(self, l2_clip, noise_multi, server_learning_rate):

        print("Creating the federated model: ")
        print("Learning rate of {}".format(server_learning_rate))

        if noise_multi is not 0:
            dp_query = tff.utils.build_dp_query(
                clip=l2_clip,
                noise_multiplier=noise_multi,
                # nr of sampled users
                expected_total_weight=10,
                # adaptive_clip_learning_rate=0,
                # target_unclipped_quantile=0.5,
                # clipped_count_budget_allocation=0.1,
                # Fraction of privacy budget to allocate for clipped counts.
                expected_clients_per_round=10)
            weights_type = tff.learning.framework.weights_type_from_model(model_fn)
            aggregation_process = tff.utils.build_dp_aggregate_process(
                weights_type.trainable, dp_query)

            # Creating Federated Model build spec
            iterative_process = tff.learning.build_federated_averaging_process(
                model_fn,
                client_optimizer_fn=lambda: tf.keras.optimizers.Adam(),
                server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=server_learning_rate),
                aggregation_process=aggregation_process
            )
        else:
            # Creating Federated Model build spec
            iterative_process = tff.learning.build_federated_averaging_process(
                model_fn,
                client_optimizer_fn=lambda: tf.keras.optimizers.Adam(),
                server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=server_learning_rate),
            )

        return iterative_process

    @staticmethod
    def fast_plot(all_mains, all_meters):

        t = pd.DataFrame(all_mains)
        plt.figure(0)
        t.plot()
        t2 = pd.DataFrame(all_meters)
        plt.figure(1)
        t2.plot()
        plt.show()

    @staticmethod
    def preprocess_nilmtk_to_np(houses_main_list, sequence_length, **load_kwargs):

        n_users = len(houses_main_list)

        print("--- Pre-processing only main data sets ---")
        print("Number of houses: {}".format(n_users))

        all_mains = np.empty(0)
        all_appliances = np.empty(0)

        for i in range(n_users):

            # load_kwargs.setdefault('sections', houses_main_list[0][0].good_sections())
            if i is 1:
                continue

            print("--- Pre-processing house {} main data set ---".format(i))
            main_power_series = houses_main_list[i][0].power_series(**load_kwargs)
            appliance_power_series = houses_main_list[i][1].power_series(**load_kwargs)

            run = True
            main_chunk = next(main_power_series)
            appliance_chunk = next(appliance_power_series)

            while run:

                # main_chunk = main_chunk[main_chunk < 1200]
                # appliance_chunk = appliance_chunk[appliance_chunk < 500]
                main_chunk = main_chunk[main_chunk >= 1]
                appliance_chunk = appliance_chunk[appliance_chunk >= 1]

                # Remove NaNs instead of filling them with 0s - Non existent data is not 0
                main_chunk.dropna(inplace=True)
                appliance_chunk.dropna(inplace=True)

                ix = main_chunk.index.intersection(appliance_chunk.index)
                main_chunk = main_chunk[ix]
                appliance_chunk = appliance_chunk[ix]

                all_mains = np.append(all_mains, main_chunk)
                all_appliances = np.append(all_appliances, appliance_chunk)

                try:
                    main_chunk = next(main_power_series)
                    appliance_chunk = next(appliance_power_series)
                except:
                    run = False

        data_points_size = len(all_mains)

        # Clipping data to fit model input
        s = sequence_length
        additional = data_points_size % s

        x_data = all_mains[:data_points_size - additional]
        y_data = all_appliances[:data_points_size - additional]

        print(" -> Number of data points from all houses mains: {}".format(x_data.size))

        return x_data, y_data

    # TODO consider creating dummy elementspec
    @staticmethod
    def get_input_spec(sequence_len, input_sample, batch_size):

        input_sample = input_sample['redd.h5 house 1']

        main_chunk = input_sample[0]
        meter_chunk = input_sample[1]

        main_chunk, meter_chunk = round_data_to_model_input(sequence_len,
                                                            main_chunk,
                                                            meter_chunk)

        x_batch = np.reshape(main_chunk, (int(len(main_chunk) / sequence_len), sequence_len, 1))
        y_batch = np.reshape(meter_chunk, (int(len(meter_chunk) / sequence_len), sequence_len, 1))

        temp_tensor_dt = tf.data.Dataset.from_tensor_slices({"x": x_batch, "y": y_batch})

        element_spec = temp_tensor_dt.batch(batch_size).map(element_spec_dict).element_spec

        return element_spec

    @staticmethod
    def evaluate_model(test_results, train_results, round_nr):

        from sklearn.metrics import roc_curve
        from scipy import stats

        start = time.time()

        # Computing ep spent
        ep_spent = compute_epsilon(round_nr, 0.33, 10/1000, 1e-2)

        # global global_input_spec
        # global_input_spec = element_spec

        loss_train = train_results[2]
        loss_test = test_results[2]
        all_loss = np.append(loss_train, loss_test)

        total_train_loss = np.sum(loss_train)/len(loss_train)
        total_test_loss = np.sum(loss_test)/len(loss_test)

        # yeom_membership_inference
        train_pred_membership_ = np.where(all_loss <= total_train_loss, 1, 0)

        membership = [np.ones(loss_train.shape[0]), np.zeros(loss_test.shape[0])]
        membership = np.concatenate(membership)

        pred_membership = np.where(
            stats.norm(0, total_test_loss).pdf(all_loss) >= stats.norm(0, total_test_loss).pdf(all_loss), 1, 0)

        fpr, tpr, thresholds = roc_curve(membership, train_pred_membership_, pos_label=1)
        yeom_mem_adv = tpr[1] - fpr[1]

        membership_ratio = yeom_mem_adv

        print("Time to generate evaluation metrics for train and test sets = {}".format(time.time() - start))

        return membership_ratio, total_train_loss, total_test_loss, ep_spent

    def fed_eval(self, test_data, train_data):
        evaluation = tff.learning.build_federated_evaluation(model_fn)
        total_train_metrics = evaluation(self.fed_state.model, train_data)
        total_train_loss = total_train_metrics['loss']
        total_test_metrics = evaluation(self.fed_state.model, test_data)
        total_test_loss = total_test_metrics['loss']
        return total_test_loss, total_train_loss

    def disaggregate_in_mem(self, main_dt, appliance_dt):

        print("--- Disaggregating dataset of {} data points ---".format(len(main_dt)))

        # Pred processing model input
        s = self.mdl_in_len
        main_dt_len = int(len(main_dt) / 10)
        main_sample = main_dt[:main_dt_len]

        mdl_input = self.in_reshape(self.mdl_in_len, main_sample)

        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
            pred = self.trained_mdl.predict(mdl_input)

        pred = np.reshape(pred, len(pred))

        # Clipping the main and gt to match prediction
        appliance_dt = appliance_dt[:main_dt_len]
        appliance_dt = appliance_dt[int(s/2):-int(s/2)]
        main_sample = main_sample[int(s/2):-int(s/2)]

        from tensorflow.keras.losses import huber
        from tensorflow.keras import backend as K

        y_true = K.variable(np.reshape(appliance_dt, (len(appliance_dt), 1)))
        y_pred = K.variable(np.reshape(pred, (len(pred), 1)))
        point_wise_loss = K.eval(huber(y_true, y_pred, delta=4))

        pred = self.data_postprocess(pred)
        main_dt = self.data_postprocess(main_sample)

        pred[pred < 0] = 0

        pred_column = pd.Series(pred, name='Fridge prediction')
        pred_powers_dict = {'Fridge prediction': pred_column}
        pred_powers = pd.DataFrame(pred_powers_dict)

        test_appliances_gt = appliance_dt
        test_appliances_gt = self.data_postprocess(test_appliances_gt)
        gt_column = pd.Series(test_appliances_gt, name='Fridge ground truth')
        gt_powers_dict = {'Fridge ground truth': gt_column}
        gt_powers = pd.DataFrame(gt_powers_dict)

        return pred_powers, gt_powers, main_dt, point_wise_loss

    def disaggregate_test(self, state, main_dt, appli_dt, loss_per_round, round_num):

        # Getting references to the last Fed state and trained model
        keras_model = gl_mdl_builder(gl_mdl_in_len)
        keras_model.compile(loss=gl_loss_func)
        state.model.assign_weights_to(keras_model)
        self.trained_mdl = keras_model

        # Pred processing model input
        s = self.mdl_in_len
        main_dt_len = int(len(main_dt) / 100)
        main_sample = main_dt[:main_dt_len]

        mdl_input = self.in_reshape(self.mdl_in_len, main_sample)

        pred = self.trained_mdl.predict(mdl_input)

        pred = np.reshape(pred, len(pred))

        pred = self.data_postprocess(pred)
        appli_dt = self.data_postprocess(appli_dt)
        main_sample = self.data_postprocess(main_sample)

        pred[pred < 0] = 0

        appli_dt = appli_dt[:main_dt_len]
        appli_dt_clip = appli_dt[int(s/2):-int(s/2)]

        rpaf = metrics.recall_precision_accuracy_f1_v2(pred, appli_dt_clip, self.on_threshold)
        rel_error = metrics.relative_error_total_energy(pred, appli_dt_clip)
        ma_error = metrics.mean_absolute_error(pred, appli_dt_clip)

        all_results = [rpaf[0], rpaf[1], rpaf[2], rpaf[3], rel_error, ma_error]

        appli_dt = appli_dt[:len(pred)]
        main_sample = main_sample[:len(pred)]

        fast_plot(main_sample,
                  pred,
                  self.appli_name,
                  power_threshold_cut=5000,
                  loss_per_round=loss_per_round,
                  all_results=all_results,
                  ground_thruth_dt=appli_dt,
                  round_num=round_num)

    def import_model(self, filename, ele_spec, l2_clip=1.0, noise_multi=0.1, micro_batches=1, learning_rate=1):
        '''Loads keras model from h5
        Parameters
        ----------
        filename : filename for .h5 file
        Returns: Keras model
        '''
        # self.model = load_model(filename)
        # with h5py.File(filename, 'a') as hf:
        #     ds = hf.get('disaggregator-data').get('mmax')
        #     self.mmax = np.array(ds)[0]

        global global_input_spec
        global_input_spec = ele_spec

        # Creating Federated Model build spec
        iterative_process = tff.learning.build_federated_averaging_process(
            model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.Adam())

        # First round of updates for the fed model
        state = iterative_process.initialize()

        ckpt_manager = FileCheckpointManager("./")
        # Index zero because this things comes as a tuple
        self.fed_state = ckpt_manager.load_latest_checkpoint(state)[0]

        keras_model = create_model(256)
        keras_model.compile(
            loss=tf.keras.losses.MeanSquaredError())

        self.fed_state.model.assign_weights_to(keras_model)
        self.trained_mdl = keras_model

    def load_from_ckpt(self, state, ckpt_number):

        ckpt_manager = FileCheckpointManager(self.ckp_dir_name)
        state = ckpt_manager.load_checkpoint(state, ckpt_number)

        return state

    def import_model_from_ckpt(self, ckpt_number, ele_spec, noise=0, l2_clip=1, server_lea=1):

        global global_input_spec
        global_input_spec = ele_spec

        iterative_process = self.build_fed_avg(l2_clip, noise, server_lea)

        # First round of updates for the fed model
        state = iterative_process.initialize()

        ckpt_manager = FileCheckpointManager('./')
        # Index zero because this things comes as a tuple
        self.fed_state = ckpt_manager.load_checkpoint(state, ckpt_number)

        keras_model = gl_mdl_builder(self.mdl_in_len)
        keras_model.compile(loss=gl_loss_func)

        self.fed_state.model.assign_weights_to(keras_model)
        self.trained_mdl = keras_model

    def export_model(self, filename):
        '''Saves keras model to h5

        Parameters
        ----------
        filename : filename for .h5 file
        '''
        self.trained_mdl.save(filename)
        with h5py.File(filename, 'a') as hf:
            gr = hf.create_group('disaggregator-data')
            gr.create_dataset('mmax', data = [self.mmax])

