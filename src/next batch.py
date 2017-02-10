import numpy as np

def next_batch(tab, size):
    batch = np.zeros(size)
    for i in range(size):
        batch[i] = tab(np.floor(np.random*size))
    return batch
