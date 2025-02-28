import tensorstore as ts
import numpy as np


def ts_read_matlab(file_path, i):
    """Reads a Zarr file using TensorStore and returns a NumPy array."""
    im_ts = ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': f'{file_path}'
        }
    }).result()

    array_data = im_ts[:, :, :, i].read().result()
    # permute to yxz for Matlab
    array_data = np.transpose(array_data, (1, 2, 0))
    return np.ascontiguousarray(array_data)
