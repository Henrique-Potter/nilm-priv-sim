from __future__ import print_function, division
import collections

import numpy as np

import matplotlib.pyplot as plt
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

import pandas as pd
import h5py
import random
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Reshape, Dropout
from tensorflow.keras.utils import plot_model

from nilm_models.StateCheckPoint import FileCheckpointManager
from nilm_models.fed_utils import *

from nilmtk.legacy.disaggregate import Disaggregator

import tensorflow as tf
import tensorflow_federated as tff

global_input_spec = None


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


def create_nt_model(sequence_len, tr=False):
    '''Creates the Auto encoder module described in the paper
    '''
    model = Sequential()

    # 1D Conv
    model.add(Conv1D(8, 4, activation="linear", input_shape=(sequence_len, 1), padding="same", strides=1, trainable=tr))
    model.add(Flatten(trainable=tr))

    # Fully Connected Layers
    model.add(Dropout(0.2, trainable=tr))
    model.add(Dense((sequence_len - 0) * 8, activation='relu', trainable=tr))

    model.add(Dropout(0.2, trainable=tr))
    model.add(Dense(128, activation='relu', trainable=tr))

    model.add(Dropout(0.2, trainable=tr))
    model.add(Dense((sequence_len - 0) * 8, activation='relu', trainable=tr))

    model.add(Dropout(0.2, trainable=tr))

    # 1D Conv
    model.add(Reshape(((sequence_len - 0), 8), trainable=tr))
    model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1, trainable=tr))

    # model.compile(loss='mse', optimizer='adam')
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


def element_spec_dict(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
        x=tf.reshape(element['x'], [-1, 256, 1]),
        y=tf.reshape(element['y'], [-1, 256, 1]))


def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = create_model(256)
    #keras_model = create_nt_model(256, False)

    return tff.learning.from_keras_model(
        keras_model,
        input_spec=global_input_spec,
        loss=tf.keras.losses.MeanSquaredError())

#DPFedDAEDisaggregator
class FedDAEDisaggregator(Disaggregator):
    def __init__(self, sequence_length):
        '''Initialize disaggregator

        Parameters
        ----------
        sequence_length : the size of window to use on the aggregate data
        meter : a nilmtk.ElecMeter meter of the appliance to be disaggregated
        '''
        self.MODEL_NAME = "AUTOENCODER"
        self.model = None
        self.sequence_length = sequence_length
        self.MIN_CHUNK_LENGTH = sequence_length
        self.fed_state = None
        self.eps_per_round_list = None
        self.element_spec = None
        self.mmax = None

    # Trains the Fed Model
    def train(self,
              all_dt,
              epochs,
              batch_size,
              synth_users_split,
              user_sample_size,
              nr_rounds,
              check_point_rate,
              learning_rate):

        train_mains_np = all_dt[0]
        train_appliances_gt = all_dt[1]

        print(" Training dataset size is {}".format(len(train_mains_np)))
        print(" Test dataset size is {}".format(len(all_dt[2])))

        train_dt_gens, _, element_spec = create_model_dt_generators(train_mains_np,
                                                                    train_appliances_gt,
                                                                    batch_size,
                                                                    epochs,
                                                                    synth_users_split,
                                                                    self.sequence_length)

        global global_input_spec
        global_input_spec = element_spec
        print("Creating the federated model: ")
        print("Learning rate of {}".format(learning_rate))
        # Creating Federated Model build spec
        iterative_process = tff.learning.build_federated_averaging_process(
            model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.Adam(),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate),
        )

        # First round of updates for the fed model
        state = iterative_process.initialize()
        random.shuffle(train_dt_gens)

        loss_per_round = []
        # Fed Training Loop
        for round_num in range(1, nr_rounds+1):
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
            if round_num % check_point_rate is 1:
                import matplotlib.pyplot as plt

                fig = plt.figure()
                ax = plt.axes()
                ax.plot(np.arange(len(loss_per_round)), loss_per_round, color='blue', linewidth=2, linestyle='solid', label="Tr L")
                ax.legend(bbox_to_anchor=(1.105, 1), shadow=True, fancybox=True)
                plt.show()
                # Create the checkpoint Manager
                ckpt_manager = FileCheckpointManager(root_dir="./")
                # Save checkpoint for round N
                ckpt_manager.save_checkpoint(state, round_num=round_num)

        # Getting references to the last Fed state and trained model
        keras_model = create_model(256)
        keras_model.compile(loss=tf.keras.losses.MeanSquaredError())
        state.model.assign_weights_to(keras_model)
        self.model = keras_model
        self.fed_state = state

    def preprocess_users_dt(self, users_train_list, batch_size, epochs, synth_users_split, load_kwargs):

        n_users = len(users_train_list)

        users_labels_np = []

        print("--- Pre-processing data sets ---")
        print("Number of houses: {}".format(n_users))
        print("User split rate: {}".format(synth_users_split))

        all_mains = np.empty(0)
        all_meters = np.empty(0)

        for i in range(n_users):

            print("--- Pre-processing house {} data set ---".format(i))

            #load_kwargs.setdefault('sections', users_train_list[i][0].good_sections())
            main_power_series = users_train_list[i][0].power_series(**load_kwargs)
            #load_kwargs.setdefault('sections', users_train_list[i][1].good_sections())
            meter_power_series = users_train_list[i][1].power_series(**load_kwargs)
            run = True
            main_chunk = next(main_power_series)
            meter_chunk = next(meter_power_series)

            while run:

                #self.fast_plot(main_chunk, meter_chunk)
                main_chunk = main_chunk[main_chunk >= 1]
                meter_chunk = meter_chunk[meter_chunk >= 1]
                # main_chunk = main_chunk[main_chunk < 1200]
                # meter_chunk = meter_chunk[meter_chunk < 500]

                # Remove NaNs instead of filling them with 0s - Non existent data is not 0
                main_chunk.dropna(inplace=True)
                meter_chunk.dropna(inplace=True)

                ix = main_chunk.index.intersection(meter_chunk.index)
                main_chunk = main_chunk[ix]
                meter_chunk = meter_chunk[ix]

                if len(main_chunk) is 0 or len(meter_chunk) is 0:
                    continue

                all_mains = np.append(all_mains, main_chunk)
                all_meters = np.append(all_meters, meter_chunk)

                try:
                    main_chunk = next(main_power_series)
                    meter_chunk = next(meter_power_series)
                except:
                    run = False

        all_mains = np.log10(all_mains)
        all_meters = np.log10(all_meters)

        dmax = np.max(all_mains)
        # all_mains = self._normalize(all_mains, dmax)
        # all_meters = self._normalize(all_meters, dmax)

        users_processed_data, _, element_spec = self.create_model_dt_generators(all_mains,
                                                                                all_meters,
                                                                                batch_size,
                                                                                epochs,
                                                                                synth_users_split)

        return users_processed_data, users_labels_np, element_spec

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

        main_chunk, meter_chunk = FedDAEDisaggregator.round_data_to_model_input(sequence_len,
                                                                                main_chunk,
                                                                                meter_chunk)

        x_batch = np.reshape(main_chunk, (int(len(main_chunk) / sequence_len), sequence_len, 1))
        y_batch = np.reshape(meter_chunk, (int(len(meter_chunk) / sequence_len), sequence_len, 1))

        temp_tensor_dt = tf.data.Dataset.from_tensor_slices({"x": x_batch, "y": y_batch})

        element_spec = temp_tensor_dt.batch(batch_size).map(element_spec_dict).element_spec

        return element_spec

    def evaluate_model(self, all_dt_eval, round_nr, report_attacks):

        start = time.time()

        train_data = all_dt_eval[0]
        tr_np_labels = all_dt_eval[1]
        test_data = all_dt_eval[2]
        test_np_labels = all_dt_eval[3]
        element_spec = all_dt_eval[4]

        end = time.time() - start
        print("Time to create 2 dts = {}".format(end))
        # TODO Working in progress
        ep_spent = compute_epsilon(round_nr, 0.3, 10/100, 1e-6)

        global global_input_spec
        global_input_spec = element_spec

        evaluation = tff.learning.build_federated_evaluation(model_fn)

        total_train_metrics = evaluation(self.fed_state.model, train_data)
        total_train_loss = total_train_metrics['loss']

        total_test_metrics = evaluation(self.fed_state.model, test_data)
        total_test_loss = total_test_metrics['loss']

        users_train_losses = []
        users_test_losses = []

        # total_train_metrics = evaluation(self.fed_state.model, train_data)
        # total_train_loss = total_train_metrics['loss']

        if report_attacks:
            fake_labels = []

            print("--> Initializing per user evaluation")
            print("Train user dt size:{}".format(len(train_data)))
            print("Test user dt size:{}".format(len(test_data)))

            for idx, user_train_dt in enumerate(train_data):
                print("- Processing Train data for User:{}".format(idx))
                train_metrics = evaluation(self.fed_state.model, [user_train_dt])
                users_train_losses.append(train_metrics['loss'])
                fake_labels.append(0)

            for idx, user_test_dt in enumerate(test_data):
                print("- Processing Test data for User:{}".format(idx))
                test_metrics = evaluation(self.fed_state.model, [user_test_dt])
                users_test_losses.append(test_metrics['loss'])

            from tensorflow_privacy.privacy.membership_inference_attack import membership_inference_attack_new as mia
            from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData

            attacks_result = mia.run_attacks(
                AttackInputData(
                    loss_train=np.array(users_train_losses),
                    loss_test=np.array(users_test_losses),
                    labels_train=np.array(fake_labels),
                    labels_test=np.array(fake_labels)
                    ))
        else:
            attacks_result = None

        return attacks_result, total_train_loss, total_test_loss, ep_spent

    def disaggregate(self, mains, output_datastore, meter_metadata, **load_kwargs):
        '''Disaggregate mains according to the model learnt.

        Parameters
        ----------
        mains : nilmtk.ElecMeter
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        meter_metadata : metadata for the produced output
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''

        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 1)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        total_model_input_lenght = 0
        total_model_output_lenght = 0
        pred_df = pd.DataFrame()

        for chunk in mains.power_series(**load_kwargs):
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue

            total_model_input_lenght += len(chunk)

            timeframes.append(chunk.timeframe)
            measurement = chunk.name
            chunk2 = self._normalize(chunk, self.mmax)

            appliance_power = self.disaggregate_chunk(chunk2)
            appliance_power[appliance_power < 0] = 0
            appliance_power = self._denormalize(appliance_power, self.mmax)

            total_model_output_lenght += len(appliance_power)

            # Append prediction to output
            data_is_available = True
            cols = pd.MultiIndex.from_tuples([chunk.name])
            meter_instance = meter_metadata.instance()
            df = pd.DataFrame(
                appliance_power.values, index=appliance_power.index,
                columns=cols, dtype="float32")
            key = '{}/elec/meter{}'.format(building_path, meter_instance)
            output_datastore.append(key, df)

            # Append aggregate data to output
            mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
            output_datastore.append(key=mains_data_location, value=mains_df)

            pred_df = pd.concat([pred_df, df])

        # Save metadata to output
        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter_metadata]
            )

        return pred_df

    def disaggregate_in_mem(self, main_dt, appliance_dt, test_split_rate=0.7):

        print("--- Merging all house main for in-mem disaggregation---")
        print("--- Using last {}---".format(test_split_rate))

        test_mains_np = main_dt
        test_appliances_gt = appliance_dt
        #_, _, test_mains_np, test_appliances_gt = self.merge_split_dts(all_db_houses, test_split_rate)

        s = self.sequence_length
        up_limit = len(test_mains_np)

        mode_input = np.reshape(test_mains_np, (int(len(test_mains_np) / s), s, 1))

        pred = self.model.predict(mode_input)

        pred = np.reshape(pred, up_limit)[:up_limit]
        pred = denormalize(pred, self.mmax)
        pred = 10**pred

        main_dt = denormalize(main_dt, self.mmax)
        main_dt = 10**main_dt

        # pred[pred < 0] = 0

        pred_column = pd.Series(pred, name='fridge')
        pred_powers_dict = {'fridge': pred_column}
        pred_powers = pd.DataFrame(pred_powers_dict)

        test_appliances_gt = denormalize(test_appliances_gt, self.mmax)
        test_appliances_gt = 10**test_appliances_gt
        gt_column = pd.Series(test_appliances_gt, name='fridge')
        gt_powers_dict = {'fridge': gt_column}
        gt_powers = pd.DataFrame(gt_powers_dict)

        return pred_powers, gt_powers, main_dt


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
        self.model = keras_model

    def import_model_from_ckpt(self, ckpt_number, ele_spec):

        global global_input_spec
        global_input_spec = ele_spec

        # Creating Federated Model build spec
        iterative_process = tff.learning.build_federated_averaging_process(
            model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.Adam())

        # First round of updates for the fed model
        state = iterative_process.initialize()

        ckpt_manager = FileCheckpointManager('./')
        # Index zero because this things comes as a tuple
        self.fed_state = ckpt_manager.load_checkpoint(state, ckpt_number)

        keras_model = create_model(256)
        keras_model.compile(
            loss=tf.keras.losses.MeanSquaredError())

        self.fed_state.model.assign_weights_to(keras_model)
        self.model = keras_model

    def export_model(self, filename):
        '''Saves keras model to h5

        Parameters
        ----------
        filename : filename for .h5 file
        '''
        self.model.save(filename)
        with h5py.File(filename, 'a') as hf:
            gr = hf.create_group('disaggregator-data')
            gr.create_dataset('mmax', data = [self.mmax])




