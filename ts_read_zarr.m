function im = ts_read_zarr(fileName, chunk_i, timepoint_i, channel_i, output_zarr_version, python_path)

if nargin < 3
    python_path = '~/miniconda3/envs/mlDataPipeline/bin/python';
end

pyenv('Version', python_path);
%disp(['Using Python: ', pe.Executable]);

%zarr_file = "/clusterfs/vast/matthewmueller/dataTest/2025/2/26/PetaKit5D_demo_cell_image_dataset/0.zarr";

ts_reader = py.importlib.import_module('ts_read');

zarr_data_py = ts_reader.ts_read_matlab(fileName, uint16(chunk_i), uint16(timepoint_i), uint16(channel_i), output_zarr_version);

im = uint16(zarr_data_py);

end


