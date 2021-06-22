from nilmtk.dataset_converters import convert_redd
from pathlib import Path
from nilmtk import DataSet
from nilm_models.dae.daedisaggregator import DAEDisaggregator
from nilmtk.datastore import HDFDataStore

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


utility_appliance = 'fridge'
sequence_length = 256

cwd = Path.cwd()
dataset_path = '..\\experiments\\data\\low_freq'
full_path = cwd.joinpath(dataset_path)

if not Path(r'..\\experiments\\data\\redd.h5').exists():
    convert_redd(str(full_path), r'..\\experiments\\data\\redd.h5')

house_models = []


#if not Path('model-redd100.h5').exists():

redd = DataSet(r'..\\experiments\\data\\redd.h5')

redd.set_window(end="30-4-2011") #Use data only until 4/30/2011

total_buildings = len(redd.buildings)
total_buildings = 1

tm_metadata_pointer = None

for i in range(total_buildings):

    house_model = DAEDisaggregator(sequence_length)
    train_elec = redd.buildings[i].elec

    train_mains = train_elec.mains().all_meters()[0] # The aggregated meter that provides the input
    train_appliance_meter = train_elec.submeters()[utility_appliance] # The microwave meter that is used as a training target
    # Used later for metadata structure
    tm_metadata_pointer = train_appliance_meter

    if not Path("house_{}_model-redd100.h5".format(i).format()).exists():
        house_model.train(train_mains, train_appliance_meter, epochs=25, sample_period=1)
        house_model.export_model("house_{}_model-redd100.h5".format(i))
    else:
        house_model.import_model("house_{}_model-redd100.h5".format(i))

    house_models.append(house_model)


test = DataSet(r'..\\experiments\\data\\redd.h5')
test.set_window(start="30-4-2011")# Use data from 4/30/2011 onward

for i in range(total_buildings):
    disag_file_path = "house_{}_disag-out.h5".format(i)
    if not Path(disag_file_path).exists():
        house_model = house_models[i]

        house_test_elec = test.buildings[i].elec
        house_test_mains = house_test_elec.mains().all_meters()[0]

        output = HDFDataStore(disag_file_path, 'w')
        # test_mains: The aggregated signal meter
        # output: The output datastore
        # tm_metadata_pointer: This is used in order to copy the metadata of the train meter into the datastore
        pred_df1 = house_model.disaggregate(house_test_mains, output, tm_metadata_pointer, sample_period=1)
        output.close()

