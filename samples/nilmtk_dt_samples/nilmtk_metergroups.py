from nilmtk.dataset_converters import convert_redd
from pathlib import Path
from matplotlib import rcParams
import matplotlib.pyplot as plt
from nilmtk import DataSet


print(Path.cwd())

cwd = Path.cwd()
dataset_path = 'data\\low_freq'
full_path = cwd.joinpath(dataset_path)

if not Path(r'data\\redd.h5').exists():
    convert_redd(str(full_path), r'data\\redd.h5')

redd = DataSet(r'data\\redd.h5')


plt.style.use('ggplot')
rcParams['figure.figsize'] = (13, 10)

elec0 = redd.buildings[1].elec
print(elec0)

elec1 = redd.buildings[2].elec
print(elec1)

fridge = elec0.mains()
fridge.plot()
plt.show()
mcwave = elec0['microwave']
series = next(fridge.power_series())
print(series.head())

series = next(mcwave.power_series())
print(series.head())


