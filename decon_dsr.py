import argparse
import inspect
import json
import os
import re

from PyPetaKit5D import XR_decon_data_wrapper
from PyPetaKit5D import XR_deskew_rotate_data_wrapper

'''
def run_decon(data_paths, channel_patterns, psf_full_paths, cpu_config_file):

    xyPixelSize = 0.108
    dz = 0.3
    Reverse = True
    dzPSF = 0.3

    # if true, check whether image is flipped in z using the setting files
    parseSettingFile = False

    # RL method
    RLMethod = 'omw'
    # wiener filter parameter
    # alpha parameter should be adjusted based on SNR and data quality.
    # typically 0.002 - 0.01 for SNR ~20; 0.02 - 0.1 or higher for SNR ~7
    wienerAlpha = 0.005
    # OTF thresholding parameter
    OTFCumThresh = 0.88
    # true if the PSF is in skew space
    skewed = True
    # deconvolution result path string (within dataPath)
    deconPathstr = 'matlab_decon_omw'

    # background to subtract
    Background = 100
    # number of iterations
    DeconIter = 2
    # erode the edge after decon for number of pixels.
    EdgeErosion = 8
    # save as 16bit, if false, save to single
    Save16bit = True
    # use zarr file as input, if false, use tiff as input
    zarrFile = False
    saveZarr = True
    # number of cpu cores
    cpusPerTask = 24
    # use cluster computing for different images
    parseCluster = False
    if cpu_config_file:
        parseCluster = True
    masterCompute = True
    # set it to true for large files that cannot be fitted to RAM/GPU, it will
    # split the data to chunks for deconvolution
    largeFile = False
    # use GPU for deconvolution
    GPUJob = False
    # if true, save intermediate results every 5 iterations.
    debug = False
    # config file for the GPU job scheduling on GPU node
    GPUConfigFile = ''
    # if true, use Matlab runtime (for the situation without matlab license)
    mccMode = True

    XR_decon_data_wrapper(data_paths, resultDirName=deconPathstr, xyPixelSize=xyPixelSize,
                          dz=dz, reverse=Reverse, channelPatterns=channel_patterns, psfFullpaths=psf_full_paths,
                          dzPSF=dzPSF, parseSettingFile=parseSettingFile, RLMethod=RLMethod,
                          wienerAlpha=wienerAlpha, OTFCumThresh=OTFCumThresh, skewed=skewed,
                          background=Background, deconIter=DeconIter, edgeErosion=EdgeErosion,
                          save16bit=Save16bit, saveZarr=saveZarr, zarrFile=zarrFile, parseCluster=parseCluster, masterCompute=masterCompute,
                          largeFile=largeFile, GPUJob=GPUJob, debug=debug, cpusPerTask=cpusPerTask,
                          configFile=cpu_config_file, GPUConfigFile=GPUConfigFile, mccMode=mccMode)


def run_dsr(data_paths, channel_patterns, cpu_config_file):

    # run deskew if true
    deskew = True
    # run rotation if true
    rotate = True
    # use combined processing if true. Note: if you only need deskew without rotation, set both Rotate and DSRCombined
    # as false.
    DSRCombined = True
    # xy pixel size
    xyPixelSize = 0.108
    # z scan step size
    dz = 0.3
    # Skew angle
    skewAngle = 32.45
    # if true, objective scan; otherwise, sample scan
    objectiveScan = False
    # scan direction
    reverse = True

    # if true, use large scale processing pipeline (split, process, and then merge)
    largeFile = False
    # true if input is in zarr format
    zarrFile = True
    # save output as zarr if true
    saveZarr = True
    # block size to save the result
    blockSize = [256, 256, 256]
    # save output as uint16 if true
    save16bit = True

    # use slurm cluster if true, otherwise use the local machine (master job)
    parseCluster = False
    if cpu_config_file:
        parseCluster = True
    # use master job for task computing or not.
    masterCompute = True
    # if true, use Matlab runtime (for the situation without matlab license)
    mccMode = True

    XR_deskew_rotate_data_wrapper(data_paths, deskew=deskew, rotate=rotate,
                                  DSRCombined=DSRCombined, xyPixelSize=xyPixelSize, dz=dz, skewAngle=skewAngle,
                                  objectiveScan=objectiveScan, reverse=reverse, channelPatterns=channel_patterns,
                                  largeFile=largeFile, zarrFile=zarrFile, saveZarr=saveZarr, blockSize=blockSize,
                                  save16bit=save16bit, parseCluster=parseCluster, masterCompute=masterCompute,
                                  configFile=cpu_config_file, mccMode=mccMode)
'''

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-file', type=str, required=True,
                    help="Paths to the folder containing tiff files separated by comma")
    ap.add_argument('--folder-paths', type=lambda s: list(map(str, s.split(','))), required=True,
                    help="Paths to the folder containing tiff files separated by comma")
    ap.add_argument('--channel-patterns', type=lambda s: list(map(str, s.split(','))), required=True,
                    help="Channel patterns separated by comma")
    ap.add_argument("--decon", action="store_true", help="Run decon")
    ap.add_argument("--dsr", action="store_true", help="Run dsr")
    args = ap.parse_args()
    input_file = args.input_file
    folder_paths = args.folder_paths
    channel_patterns = args.channel_patterns
    decon = args.decon
    dsr = args.dsr

    with open(input_file) as f:
        datasets_json = json.load(f)

    datasets = {}
    for folder_path in folder_paths:
        datasets[folder_path] = datasets_json[folder_path]

    with open(inspect.getfile(XR_decon_data_wrapper), "r", encoding="utf-8") as f:
        decon_text = f.read()

    with open(inspect.getfile(XR_deskew_rotate_data_wrapper), "r", encoding="utf-8") as f:
        dsr_text = f.read()

    matches = re.findall(r'"(.*?)": \[', decon_text)
    valid_decon_params = {key: True for key in matches}

    matches = re.findall(r'"(.*?)": \[', dsr_text)
    valid_dsr_params = {key: True for key in matches}

    run_decon = getattr(__import__('PyPetaKit5D'), 'XR_decon_data_wrapper')
    run_dsr = getattr(__import__("PyPetaKit5D"), 'XR_deskew_rotate_data_wrapper')

    if decon:
        for folder_path, dataset in datasets.items():
            kwargs = {}
            for param, value in dataset.items():
                if param in  valid_decon_params:
                    kwargs[param] = value
            kwargs['channelPatterns'] = channel_patterns
            run_decon([folder_path], **kwargs)
    if dsr:
        for folder_path, dataset in datasets.items():
            kwargs = {}
            for param, value in dataset.items():
                if param in valid_dsr_params:
                    kwargs[param] = value
            kwargs['channelPatterns'] = channel_patterns
            if decon:
                if 'resultDirName' in dataset:
                    folder_path = os.path.join(folder_path, dataset['resultDirName'])
                else:
                    folder_path = os.path.join(folder_path, 'matlab_decon')
            run_decon([folder_path], **kwargs)
