from __future__ import print_function, division
import random
import sys
import collections

from matplotlib import rcParams
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import h5py

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Reshape, Dropout
from tensorflow.keras.utils import plot_model

from nilmtk.legacy.disaggregate import Disaggregator


import tensorflow as tf
import tensorflow_federated as tff

global global_input_spec


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
    # plot_model(model, to_file='model.png', show_shapes=True)

    return model


def loss_fn_federated(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))


def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = create_model(256)
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=global_input_spec,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.CosineSimilarity()])


class TFF_DAEDisaggregator(Disaggregator):
    '''Denoising Autoencoder disaggregator from Neural NILM
    https://arxiv.org/pdf/1507.06594.pdf

    Attributes
    ----------
    model : keras Sequential model
    sequence_length : the size of window to use on the aggregate data
    mmax : the maximum value of the aggregate data

    MIN_CHUNK_LENGTH : int
       the minimum length of an acceptable chunk
    '''

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
        #self.model = self._create_model(self.sequence_length)

    def train(self, users_train_list, epochs=25, batch_size = 16, **load_kwargs):
        '''Train

        Parameters
        ----------
        mains : a nilmtk.ElecMeter object for the aggregate data
        meter : a nilmtk.ElecMeter object for the meter data
        epochs : number of epochs to train
        **load_kwargs : keyword arguments passed to `meter.power_series()`
        '''

        n_users = len(users_train_list)
        all_users_data = []
        element_spec = None

        for i in range(1):
            main_power_series = users_train_list[i][0].power_series(**load_kwargs)
            meter_power_series = users_train_list[i][1].power_series(**load_kwargs)
            run = True
            mainchunk = next(main_power_series)
            meterchunk = next(meter_power_series)
            user_full_dataset = None

            if self.mmax == None:
                self.mmax = mainchunk.max()
            while run:

                mainchunk = self._normalize(mainchunk, self.mmax)
                meterchunk = self._normalize(meterchunk, self.mmax)

                # Replace NaNs with 0s
                mainchunk.fillna(0, inplace=True)
                meterchunk.fillna(0, inplace=True)
                ix = mainchunk.index.intersection(meterchunk.index)
                mainchunk = mainchunk[ix]
                meterchunk = meterchunk[ix]
                data_points_size = len(ix)

                s = self.sequence_length
                additional = s - (data_points_size % s)

                x_batch = np.append(mainchunk, np.zeros(additional, dtype=np.float32))
                y_batch = np.append(meterchunk, np.zeros(additional, dtype=np.float32))

                x_batch = np.reshape(x_batch, (int(len(x_batch) / s), s, 1))
                y_batch = np.reshape(y_batch, (int(len(y_batch) / s), s, 1))

                nr_users = 10
                print("--- Creating synthetic users base ---")
                print("Number of data points: {}".format(data_points_size))
                print("Number of synthetic user: {}".format(nr_users))
                print("Average user dataset size: {}".format(data_points_size/nr_users))

                split_x_batch = np.array_split(x_batch, nr_users)
                split_y_batch = np.array_split(y_batch, nr_users)

                for x_split, y_split in zip(split_x_batch, split_y_batch):

                    temp_tensor_dt = tf.data.Dataset.from_tensor_slices({"x": x_split, "y": y_split})
                    split_dt_prefetcher = TFF_DAEDisaggregator.preprocess(temp_tensor_dt)
                    element_spec = split_dt_prefetcher.element_spec
                    all_users_data.append(split_dt_prefetcher)

                    # #user_full_dataset = user_full_dataset.concatenate(pre_processed)
                    # all_users_data.append(pre_processed)

                try:
                    mainchunk = next(main_power_series)
                    meterchunk = next(meter_power_series)
                except:
                    run = False

        global global_input_spec
        global_input_spec = element_spec

        iterative_process = tff.learning.build_federated_averaging_process(
            model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.Adam(),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

        state = iterative_process.initialize()

        state, metrics = iterative_process.next(state, all_users_data)

        print('round  1, metrics={}'.format(metrics))

        NUM_ROUNDS = 15
        for round_num in range(2, NUM_ROUNDS):
            state, metrics = iterative_process.next(state, all_users_data)
            print('round {:2d}, metrics={}'.format(round_num, metrics))

        keras_model = create_model(256)
        keras_model.compile(
            loss=tf.keras.losses.MeanSquaredError())

        state.model.assign_weights_to(keras_model)
        self.model = keras_model

        #self.train_on_chunk(users_chunk_list, epochs, batch_size)

    @staticmethod
    def preprocess(dataset, epochs=25, batch_size=16, prefetch=1000):
        def batch_format_fn(element):
            """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
            return collections.OrderedDict(
                x=tf.reshape(element['x'], [-1, 256, 1]),
                y=tf.reshape(element['y'], [-1, 256, 1]))

        return dataset.repeat(epochs).shuffle(100).batch(batch_size).map(batch_format_fn).prefetch(prefetch)

    def train_on_chunk(self, users_chunk_list, epochs, batch_size):
        '''Train using only one chunk

        Parameters
        ----------
        mainchunk : chunk of site meter
        meterchunk : chunk of appliance
        epochs : number of epochs for training
        '''

        #self.model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=epochs, shuffle=True)

    def train_across_buildings(self, mainlist, meterlist, epochs=1, batch_size=128, **load_kwargs):
        assert len(mainlist) == len(meterlist), "Number of main and meter channels should be equal"
        num_meters = len(mainlist)

        mainps = [None] * num_meters
        meterps = [None] * num_meters
        mainchunks = [None] * num_meters
        meterchunks = [None] * num_meters

        for i,m in enumerate(mainlist):
            mainps[i] = m.power_series(**load_kwargs)

        for i,m in enumerate(meterlist):
            meterps[i] = m.power_series(**load_kwargs)

        for i in range(num_meters):
            mainchunks[i] = next(mainps[i])
            meterchunks[i] = next(meterps[i])
        if self.mmax == None:
            self.mmax = max([m.max() for m in mainchunks])


        run = True
        while(run):
            mainchunks = [self._normalize(m, self.mmax) for m in mainchunks]
            meterchunks = [self._normalize(m, self.mmax) for m in meterchunks]

            self.train_across_buildings_chunk(mainchunks, meterchunks, epochs, batch_size)
            try:
                for i in range(num_meters):
                    mainchunks[i] = next(mainps[i])
                    meterchunks[i] = next(meterps[i])
            except:
                run = False

    def train_across_buildings_chunk(self, mainchunks, meterchunks, epochs, batch_size):
        num_meters = len(mainchunks)
        batch_size = int(batch_size/num_meters)
        num_of_batches = [None] * num_meters
        s = self.sequence_length
        for i in range(num_meters):
            mainchunks[i].fillna(0, inplace=True)
            meterchunks[i].fillna(0, inplace=True)
            ix = mainchunks[i].index.intersection(meterchunks[i].index)
            m1 = mainchunks[i]
            m2 = meterchunks[i]
            mainchunks[i] = m1[ix]
            meterchunks[i] = m2[ix]

            num_of_batches[i] = int(len(ix)/(s*batch_size)) - 1

        for e in range(epochs):
            print(e)
            batch_indexes = list(range(min(num_of_batches)))
            random.shuffle(batch_indexes)

            for bi, b in enumerate(batch_indexes):

                print("Batch {} of {}".format(bi,num_of_batches), end="\r")
                sys.stdout.flush()
                X_batch = np.empty((batch_size*num_meters, s, 1))
                Y_batch = np.empty((batch_size*num_meters, s, 1))

                for i in range(num_meters):
                    mainpart = mainchunks[i]
                    meterpart = meterchunks[i]
                    mainpart = mainpart[b*batch_size*s:(b+1)*batch_size*s]
                    meterpart = meterpart[b*batch_size*s:(b+1)*batch_size*s]
                    X = np.reshape(mainpart, (batch_size, s, 1))
                    Y = np.reshape(meterpart, (batch_size, s, 1))

                    X_batch[i*batch_size:(i+1)*batch_size] = np.array(X)
                    Y_batch[i*batch_size:(i+1)*batch_size] = np.array(Y)

                p = np.random.permutation(len(X_batch))
                X_batch, Y_batch = X_batch[p], Y_batch[p]

                self.model.train_on_batch(X_batch, Y_batch)
            print("\n")

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
            print("New sensible chunk hdf: {}".format(len(chunk)))

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

        print("Model input size: {}".format(total_model_input_lenght))
        print("Model output size: {}".format(total_model_output_lenght))


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

            measurement = chunk.name
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

    def import_model(self, filename):
        '''Loads keras model from h5

        Parameters
        ----------
        filename : filename for .h5 file

        Returns: Keras model
        '''
        self.model = load_model(filename)
        with h5py.File(filename, 'a') as hf:
            ds = hf.get('disaggregator-data').get('mmax')
            self.mmax = np.array(ds)[0]

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


