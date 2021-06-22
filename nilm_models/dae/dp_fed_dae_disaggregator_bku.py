from __future__ import print_function, division
import collections

import numpy as np


from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

import pandas as pd
import h5py
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Reshape, Dropout
from tensorflow.keras.utils import plot_model

from nilm_models.StateCheckPoint import FileCheckpointManager
from nilm_models.fed_utils import compute_epsilon

from nilmtk.legacy.disaggregate import Disaggregator

import tensorflow as tf
import tensorflow_federated as tff

global_input_spec = None

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def create_dp_optimizer(l2_clip, noise_multi, micro_batches, learning_rate):

    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=l2_clip,
        noise_multiplier=noise_multi,
        num_microbatches=micro_batches,
        learning_rate=learning_rate)

    return optimizer


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
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=global_input_spec,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.CosineSimilarity()])


class DPFedDAEDisaggregator(Disaggregator):
    def __init__(self, sequence_length):
        '''Initialize disaggregator

        Parameters
        ----------
        sequence_length : the size of window to use on the aggregate data
        meter : a nilmtk.ElecMeter meter of the appliance to be disaggregated
        '''
        self.MODEL_NAME = "AUTOENCODER"
        self.mmax = None
        self.model = None
        self.sequence_length = sequence_length
        self.MIN_CHUNK_LENGTH = sequence_length
        self.fed_state = None
        self.eps_per_round_list = None
        self.element_spec = None

    # Trains the Fed Model
    def train(self, users_train_list,
              epochs=25,
              batch_size=16,
              synth_users_split=100,
              user_sample_size=10,
              nr_rounds=10,
              l2_clip=1.0,
              noise_multi=0.1,
              micro_batches=1,
              learning_rate=1, **load_kwargs):

        all_users_data, total_users, _, element_spec = self.create_users_dt(users_train_list,
                                                                            batch_size,
                                                                            epochs,
                                                                            synth_users_split,
                                                                            load_kwargs)

        global global_input_spec
        global_input_spec = element_spec

        # Creating Federated Model build spec
        iterative_process = tff.learning.build_federated_averaging_process(
            model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.Adam(),
            server_optimizer_fn=lambda: create_dp_optimizer(l2_clip, noise_multi, micro_batches, learning_rate))

        # First round of updates for the fed model
        state = iterative_process.initialize()

        # Fed Training Loop
        for round_num in range(1, nr_rounds+1):
            # Random sub sample of users
            selected_users = random.sample(all_users_data, user_sample_size)
            # 1 Round of client training and fed model update
            state, metrics = iterative_process.next(state, selected_users)
            print('round {:2d}, metrics={}'.format(round_num, metrics))

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

    def create_users_dt(self, users_train_list, batch_size, epochs, synth_users_split, load_kwargs):

        element_spec = None
        users_processed_data = []
        n_users = len(users_train_list)
        total_users = n_users * synth_users_split

        users_labels_np = []

        print("--- Pre-processing data sets ---")
        print("Number of houses: {}".format(n_users))
        print("User split rate: {}".format(synth_users_split))
        print("Total number of users (with users split): {}".format(total_users))
        for i in range(n_users):

            print("--- Pre-processing house {} data set ---".format(i))

            main_power_series = users_train_list[i][0].power_series(**load_kwargs)
            meter_power_series = users_train_list[i][1].power_series(**load_kwargs)
            run = True
            main_chunk = next(main_power_series)
            meter_chunk = next(meter_power_series)

            if self.mmax is None:
                self.mmax = main_chunk.max()

            while run:

                main_chunk = self._normalize(main_chunk, self.mmax)
                meter_chunk = self._normalize(meter_chunk, self.mmax)

                # Replace NaNs with 0s
                main_chunk.fillna(0, inplace=True)
                meter_chunk.fillna(0, inplace=True)
                ix = main_chunk.index.intersection(meter_chunk.index)
                main_chunk = main_chunk[ix]
                meter_chunk = meter_chunk[ix]
                data_points_size = len(ix)

                print("Number of data points in this chunk: {}".format(data_points_size))
                print("Average user dataset size: {}".format(int(data_points_size / synth_users_split)))

                s = self.sequence_length
                # Create array of batches
                # additional = s - ((up_limit-down_limit) % s)
                additional = s - (data_points_size % s)

                x_batch = np.append(main_chunk, np.zeros(additional, dtype=np.float32))
                y_batch = np.append(meter_chunk, np.zeros(additional, dtype=np.float32))

                x_batch = np.reshape(x_batch, (int(len(x_batch) / s), s, 1))
                y_batch = np.reshape(y_batch, (int(len(y_batch) / s), s, 1))

                split_x_batch = np.array_split(x_batch, synth_users_split)
                split_y_batch = np.array_split(y_batch, synth_users_split)

                for x_split, y_split in zip(split_x_batch, split_y_batch):
                    temp_tensor_dt = tf.data.Dataset.from_tensor_slices({"x": x_split, "y": y_split})
                    split_dt_prefetcher = DPFedDAEDisaggregator.preprocess(temp_tensor_dt, epochs, batch_size)
                    element_spec = split_dt_prefetcher.element_spec
                    users_processed_data.append(split_dt_prefetcher)
                    users_labels_np.append(y_split)
                try:
                    main_chunk = next(main_power_series)
                    meter_chunk = next(meter_power_series)
                except:
                    run = False

        return users_processed_data, total_users, users_labels_np, element_spec

    @staticmethod
    def get_input_spec(s, users_train_list, batch_size, **load_kwargs):

        main_power_series = users_train_list[0][0].power_series(**load_kwargs)
        meter_power_series = users_train_list[0][1].power_series(**load_kwargs)
        main_chunk = next(main_power_series)
        meter_chunk = next(meter_power_series)

        # Replace NaNs with 0s
        main_chunk.fillna(0, inplace=True)
        meter_chunk.fillna(0, inplace=True)
        ix = main_chunk.index.intersection(meter_chunk.index)
        main_chunk = main_chunk[ix]
        meter_chunk = meter_chunk[ix]
        data_points_size = len(ix)

        additional = s - (data_points_size % s)

        x_batch = np.append(main_chunk, np.zeros(additional, dtype=np.float32))
        y_batch = np.append(meter_chunk, np.zeros(additional, dtype=np.float32))

        x_batch = np.reshape(x_batch, (int(len(x_batch) / s), s, 1))
        y_batch = np.reshape(y_batch, (int(len(y_batch) / s), s, 1))

        temp_tensor_dt = tf.data.Dataset.from_tensor_slices({"x": x_batch, "y": y_batch})

        element_spec = temp_tensor_dt.batch(batch_size).map(element_spec_dict).element_spec

        return element_spec

    def evaluate_model(self,
                       users_train_data,
                       users_test_data,
                       batch_size,
                       epochs,
                       synth_users_split,
                       **load_kwargs):

        train_data, dt_size, tr_np_labels, element_spec = self.create_users_dt(users_train_data,
                                                                         batch_size,
                                                                         epochs,
                                                                         synth_users_split,
                                                                         load_kwargs)

        test_data, _, test_np_labels, element_spec = self.create_users_dt(users_test_data,
                                                                          batch_size,
                                                                          epochs,
                                                                          synth_users_split,
                                                                          load_kwargs)

        # # TODO Verify if it makes sense, likely it does not
        # eps = compute_epsilon(epochs * round_num // batch_size, noise_multi, batch_size, total_users * 256)
        # self.eps_per_round_list.append(eps)
        # print('For delta=1e-4, the current epsilon is: %.2f' % eps)

        global global_input_spec
        global_input_spec = element_spec

        evaluation = tff.learning.build_federated_evaluation(model_fn)
        print(str(evaluation.type_signature))

        users_train_losses = []
        users_test_losses = []

        # train_metrics = evaluation(self.fed_state.model, train_data)
        # users_train_losses.append(train_metrics)

        fake_labels = []

        print("--> Initializing per user evaluation")
        print("--> Train user dt size:{}".format(len(train_data)))
        print("--> Test user dt size:{}".format(len(test_data)))

        for idx, user_train_dt in enumerate(train_data):
            print("--> Processing Train data for User:{}".format(idx))
            train_metrics = evaluation(self.fed_state.model, [user_train_dt])
            users_train_losses.append(train_metrics['loss'])
            fake_labels.append(0)

        for idx, user_test_dt in enumerate(test_data):
            print("--> Processing Test data for User:{}".format(idx))
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

        print(attacks_result.summary())

        # steps = 1
        # noise = 0.1
        # ep_batch = 10
        # data_base_size = len(train_data)
        # delta = 1e-6
        # ep_spent = compute_epsilon(steps, noise, ep_batch, data_base_size, delta)

        return attacks_result

    @staticmethod
    def preprocess(dataset, epochs, batch_size, prefetch=10):
        def batch_format_fn(element):
            """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
            return collections.OrderedDict(
                x=tf.reshape(element['x'], [-1, 256, 1]),
                y=tf.reshape(element['y'], [-1, 256, 1]))

        return dataset.repeat(epochs).shuffle(10000).batch(batch_size).map(batch_format_fn).prefetch(prefetch)

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

    def disaggregate_in_mem(self, mains, **load_kwargs):
        '''Disaggregate mains according to the model learnt.

        Parameters
        ----------
        mains : nilmtk.ElecMeter
        meter_metadata : metadata for the produced output
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''

        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 1)
        load_kwargs.setdefault('sections', mains.good_sections())

        pred_df = pd.DataFrame()

        total_model_input_len = 0
        total_model_output_len = 0

        for chunk in mains.power_series(**load_kwargs):
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue
            print("New sensible chunk in mem: {}".format(len(chunk)))

            total_model_input_len += len(chunk)

            chunk2 = self._normalize(chunk, self.mmax)

            appliance_power = self.disaggregate_chunk(chunk2)
            appliance_power[appliance_power < 0] = 0
            appliance_power = self._denormalize(appliance_power, self.mmax)

            total_model_output_len += len(appliance_power)

            cols = pd.MultiIndex.from_tuples([chunk.name])
            df = pd.DataFrame(
                appliance_power.values, index=appliance_power.index,
                columns=cols, dtype="float32")

            pred_df = pd.concat([pred_df, df])

        print("Model input size: {}".format(total_model_input_len))
        print("Model output size: {}".format(total_model_output_len))
        print("Model df output size: {}".format(len(pred_df)))

        return pred_df

    def disaggregate_chunk(self, mains):
        '''In-memory disaggregation.

        Parameters
        ----------
        mains : pd.Series to disaggregate
        Returns
        -------
        appliance_powers : pd.DataFrame where each column represents a
            disaggregated appliance.  Column names are the integer index
            into `self.model` for the appliance in question.
        '''
        s = self.sequence_length
        up_limit = len(mains)

        mains.fillna(0, inplace=True)

        additional = s - (up_limit % s)
        X_batch = np.append(mains, np.zeros(additional))
        X_batch = np.reshape(X_batch, (int(len(X_batch) / s), s, 1))

        pred = self.model.predict(X_batch)
        pred = np.reshape(pred, (up_limit + additional))[:up_limit]
        column = pd.Series(pred, index=mains.index, name=0)

        appliance_powers_dict = {}
        appliance_powers_dict[0] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict)
        return appliance_powers

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
            client_optimizer_fn=lambda: tf.keras.optimizers.Adam(),
            server_optimizer_fn=lambda: create_dp_optimizer(l2_clip, noise_multi, micro_batches, learning_rate))

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

    def import_model_from_ckpt(self, ckpt_number, ele_spec, l2_clip=1.0, noise_multi=0.1, micro_batches=1, learning_rate=1):

        # self.model = load_model(filename)
        # with h5py.File(filename, 'a') as hf:
        #     ds = hf.get('disaggregator-data').get('mmax')
        #     self.mmax = np.array(ds)[0]

        global global_input_spec
        global_input_spec = ele_spec

        # Creating Federated Model build spec
        iterative_process = tff.learning.build_federated_averaging_process(
            model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.Adam(),
            server_optimizer_fn=lambda: create_dp_optimizer(l2_clip, noise_multi, micro_batches, learning_rate))

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

    def _normalize(self, chunk, mmax):
        '''Normalizes timeseries

        Parameters
        ----------
        chunk : the timeseries to normalize
        max : max value of the powerseries

        Returns: Normalized timeseries
        '''
        tchunk = chunk / mmax
        return tchunk

    def _denormalize(self, chunk, mmax):
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


