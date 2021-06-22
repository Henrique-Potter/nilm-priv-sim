import time
from nilm_models.dae.dpfeddaedisaggregator import DPFedDAEDisaggregator as DPFedDisag
from nilm_models.fed_utils import *
from tensorflow.keras.backend import clear_session
from pathlib import Path
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYDEVD_USE_FRAME_EVAL'] = 'NO'

sequence_length = 99


def main():

    clear_session()

    warnings.simplefilter("ignore")
    import numpy as np
    #np.random.seed(0)

    # import nest_asyncio
    # nest_asyncio.apply()

    # Appliance Target
    utility_appliance = 'fridge'
    # NN Model input size
    sequence_length = 101

    h5_data_souce_path = '..\\data\\'

    users_data = Path(r'./processed_input_data.pkl').resolve()

    all_db_houses = process_nilmtk_h5(h5_data_souce_path, users_data, utility_appliance)

    test_split = 0.7
    all_dt = merge_all_houses(all_db_houses,
                              sequence_length,
                              test_split)

    # Instantiating DP FED experiment class
    fed_dp_dae = DPFedDisag(sequence_length)

    # Training model or loading from checkpoint
    start = time.time()
    fed_dp_dae.train(all_dt,
                     epochs=5,
                     batch_size=256,
                     synth_users_split=1000,
                     user_sample_size=10,
                     nr_rounds=120,
                     l2_clip=0.3,
                     noise_multi=0.3,
                     check_point_rate=1,
                     server_learning_rate=1,
                     appli_name=utility_appliance)

    end = time.time() - start
    # clear_session()
    print('--- Training completed in {} minutes---'.format(end / 60.0))


if __name__ == "__main__":
    main()
