function im = ts_read_zarr(fileName, i, python_path)

if nargin < 3
    python_path = '~/miniconda3/envs/mlDataPipeline/bin/python';
end

pyenv('Version', python_path);
%disp(['Using Python: ', pe.Executable]);

%zarr_file = "/clusterfs/vast/matthewmueller/dataTest/2025/2/26/PetaKit5D_demo_cell_image_dataset/0.zarr";

ts_reader = py.importlib.import_module('ts_read');

zarr_data_py = ts_reader.ts_read_matlab(fileName, uint16(i));

im = uint16(zarr_data_py);

end


