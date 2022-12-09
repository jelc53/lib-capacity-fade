import numpy as np
import pickle
import os


def load_data(batch1_file, batch2_file, batch3_file):
    """Read each batch and combine into signle data dictionary"""

    with open(os.path.join('data', batch1_file), 'rb') as f:
        batch1 = pickle.load(f)

    with open(os.path.join('data', batch2_file), 'rb') as f:
        batch2 = pickle.load(f)

    with open(os.path.join('data', batch3_file), 'rb') as f:
        batch3 = pickle.load(f)

    # remove batteries that do not reach 80% capacity
    del batch1['b1c8']
    del batch1['b1c10']
    del batch1['b1c12']
    del batch1['b1c13']
    del batch1['b1c22']

    # there are four cells from batch1 that carried into batch2, we'll remove the data from batch2
    # and put it with the correct cell from batch1
    batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
    batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
    add_len = [662, 981, 1060, 208, 482]

    for i, bk in enumerate(batch1_keys):
        batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
        for j in batch1[bk]['summary'].keys():
            if j == 'cycle':
                batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
            else:
                batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
        last_cycle = len(batch1[bk]['cycles'].keys())
        for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
            batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]

    # remove now duplicated channels from batch2
    del batch2['b2c7']
    del batch2['b2c8']
    del batch2['b2c9']
    del batch2['b2c15']
    del batch2['b2c16']

    # remove noisy channels from batch3
    del batch3['b3c37']
    del batch3['b3c2']
    del batch3['b3c23']
    del batch3['b3c32']
    del batch3['b3c42']
    del batch3['b3c43']

    numBat1 = len(batch1.keys())
    numBat2 = len(batch2.keys())
    numBat3 = len(batch3.keys())
    # numBat = numBat1 + numBat2 + numBat3
    bat_dict = {
        **batch1,
        **batch2,
        **batch3
    }

    return bat_dict


def remove_intracycle_data(bat_dict):
    """Delete intra cycle data dictionary from each b#c#"""
    for bc_ref in bat_dict.keys():
        del bat_dict[bc_ref]['cycles']

    return bat_dict


def write_file(data_dict, filename):
    """Write to pickle file format"""
    with open(os.path.join('data', filename), 'wb') as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    data_dict = load_data('batch1.pkl', 'batch2.pkl', 'batch3.pkl')
    lite_dict = remove_intracycle_data(bat_dict=data_dict)
    write_file(lite_dict, 'processed_data_lite.pkl')
