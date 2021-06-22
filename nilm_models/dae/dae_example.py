from nilmtk.dataset_converters import convert_redd
from pathlib import Path
from nilmtk import DataSet
from nilm_models.dae.daedisaggregator import DAEDisaggregator

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


cwd = Path.cwd()
dataset_path = '..\\experiments\\data\\low_freq'
raw_data = cwd.joinpath(dataset_path).resolve()
nilmtk_h5_path = Path(r'..\\experiments\\data\\redd.h5').resolve()

if not nilmtk_h5_path.exists():
    convert_redd(str(raw_data), nilmtk_h5_path)


dae = DAEDisaggregator(256)
#if not Path('model-redd100.h5').exists():

redd = DataSet(nilmtk_h5_path)

redd.set_window(end="30-4-2011") #Use data only until 4/30/2011
train_elec = redd.buildings[1].elec

train_mains = train_elec.mains().all_meters()[0] # The aggregated meter that provides the input
train_meter = train_elec.submeters()['fridge'] # The microwave meter that is used as a training target

if not Path("model-redd100.h5").exists():
    dae.train(train_mains, train_meter, epochs=25, sample_period=1)
    dae.export_model("model-redd100.h5")
else:
    dae.import_model("model-redd100.h5")

test = DataSet(nilmtk_h5_path)
test.set_window(start="30-4-2011") # Use data from 4/30/2011 onward
test_elec = test.buildings[1].elec
#test_elec.plot_when_on(on_power_threshold = 10)
test_mains = test_elec.mains().all_meters()[0]

disag_filename = 'disag-out.h5' # The filename of the resulting datastore
pred_df1 = None

if not Path("disag-out.h5").exists():
    from nilmtk.datastore import HDFDataStore
    output = HDFDataStore(disag_filename, 'w')
    # test_mains: The aggregated signal meter
    # output: The output datastore
    # train_meter: This is used in order to copy the metadata of the train meter into the datastore
    pred_df1 = dae.disaggregate(test_mains, output, train_meter, sample_period=1)
    output.close()


pred_df = dae.disaggregate_in_mem(test_mains, sample_period=1)


result = DataSet(disag_filename)
res_elec = result.buildings[1].elec
predicted = res_elec['fridge']
ground_truth = test_elec['fridge']


import matplotlib.pyplot as plt
ground_truth.plot()
plt.show()

# train_meter.plot()
# plt.show()
predicted.plot()
plt.show()


from nilm_models.dae import metrics

rpaf = metrics.recall_precision_accuracy_f1(predicted, ground_truth)

print("============ Recall: {}".format(rpaf[0]))
print("============ Precision: {}".format(rpaf[1]))
print("============ Accuracy: {}".format(rpaf[2]))
print("============ F1 Score: {}".format(rpaf[3]))

print("============ Relative error in total energy: {}".format(
    metrics.relative_error_total_energy(predicted, ground_truth)))
print("============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(predicted, ground_truth)))
