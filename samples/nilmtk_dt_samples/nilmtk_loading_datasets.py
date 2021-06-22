from nilmtk.dataset_converters import convert_redd
from pathlib import Path

print(Path.cwd())

cwd = Path.cwd()
dataset_path = 'data\\low_freq'
full_path = cwd.joinpath(dataset_path)

if not Path(r'data\\redd.h5').exists():
    convert_redd(str(full_path), r'data\\redd.h5')

from nilmtk import DataSet
from nilmtk.utils import print_dict

redd = DataSet(r'data\\redd.h5')

print_dict(redd.metadata)


elec = redd.buildings[1].elec
print("\n All data from building 1  ----- \n")
print(elec)

fridge = elec['fridge']
print("\n All columns available for a fridge from Building 1   ----- \n")
print(fridge.available_columns())

df = next(fridge.load())
print("\n Df Head  ----- \n")
print(df.head())

series = next(fridge.power_series())
print("\n Power series ----- \n")
print(series.head())