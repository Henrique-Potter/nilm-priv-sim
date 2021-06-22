from nilmtk.dataset_converters import convert_redd
from pathlib import Path
from nilmtk import DataSet

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


cwd = Path.cwd()
dataset_path = '..\\..\\experiments\\data\\low_freq'
full_path = cwd.joinpath(dataset_path)

if not Path(r'..\\..\\experiments\\data\\redd.h5').exists():
    convert_redd(str(full_path), r'..\\..\\experiments\\data\\redd.h5')

from nilm_models.rnn.rnndisaggregator import RNNDisaggregator

redd = DataSet(r'..\\..\\experiments\\data\\redd.h5')

redd.set_window(end="30-4-2011") #Use data only until 4/30/2011
train_elec = redd.buildings[1].elec

train_mains = train_elec.mains().all_meters()[0] # The aggregated meter that provides the input
train_meter = train_elec.submeters()['fridge']

rnn = RNNDisaggregator()

if not Path("model-redd5.h5").exists():
    rnn.train(train_mains, train_meter, epochs=5, sample_period=1)
    rnn.export_model("model-redd5.h5")
else:
    rnn.import_model("model-redd5.h5")

test = DataSet(r'..\\..\\experiments\\data\\redd.h5')
test.set_window(start="30-4-2011")
test_elec = test.buildings[1].elec
test_mains = test_elec.mains().all_meters()[0]

disag_filename = 'disag-out.h5'
pred_df1 = None
if not Path("disag-out.h5").exists():
    from nilmtk.datastore import HDFDataStore

    output = HDFDataStore(disag_filename, 'w')

    # test_mains: The aggregated signal meter
    # output: The output datastore
    # train_meter: This is used in order to copy the metadata of the train meter into the datastore
    rnn.disaggregate(test_mains, output, train_meter, sample_period=1)
    output.close()

result = DataSet(disag_filename)
res_elec = result.buildings[1].elec
predicted = res_elec['fridge']
ground_truth = train_meter['fridge']

import matplotlib.pyplot as plt
predicted.plot()
plt.show()
ground_truth.plot()
plt.show()
test_elec.mains().plot()
plt.show()
