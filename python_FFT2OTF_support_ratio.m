function python_FFT2OTF_support_ratio(fn, fn_psf_ideal, chunk_i, timepoint_i, channel_i, python_path, varargin)
    volume = ts_read_zarr(fn, chunk_i, timepoint_i, channel_i, python_path);
    psf = tiffreadVolume(fn_psf_ideal);
    numTimepoints = size(volume, 1);
    support_ratio_avg = struct();
    fields = {};
    for i = 1 : numTimepoints
        support_ratio = python_FFT2OTF_support_ratio_func(squeeze(volume(i, :, :, :)), psf, varargin{:});
        if i ==1
            fields = fieldnames(support_ratio);
            for f = 1:numel(fields)
                support_ratio_avg.(fields{f}) = 0;
            end
        end
        for f = 1:numel(fields)
            support_ratio_avg.(fields{f}) = support_ratio_avg.(fields{f}) + support_ratio.(fields{f});
        end
    end

    % Compute the average
    for f = 1:numel(fields)
        support_ratio_avg.(fields{f}) = support_ratio_avg.(fields{f}) / numTimepoints;
    end

    create_json_file(fn, chunk_i, timepoint_i, channel_i, support_ratio_avg);

end

function support_ratio = python_FFT2OTF_support_ratio_func(volume, psf, varargin)
% calculate the ratio fo OTF support

% Parse inputs
ip = inputParser;
ip.CaseSensitive = false;
ip.addRequired('volume');
ip.addRequired('psf');
ip.addParameter('xyPixelSize', 0.108, @isnumeric);
ip.addParameter('dz', 0.2, @isnumeric);
ip.addParameter('outPixelSize', [.0367 .0367 .0367] , @isnumeric);
ip.addParameter('N', [349, 349, 349] , @isnumeric);
ip.addParameter('interpMethod', 'linear' );
ip.addParameter('wienerAlpha', 0.005, @isnumeric);
ip.addParameter('OTFCumThresh', 0.95, @isnumeric);
ip.addParameter('minIntThrsh', 1e-3, @isnumeric);
ip.addParameter('OTFAreaThresh', 200, @isnumeric);
ip.addParameter('skewed', false, @islogical);
ip.parse(volume, psf, varargin{:});
pr = ip.Results;

xyPixelSize = pr.xyPixelSize;
dz = pr.dz;
N = pr.N;
interpMethod = pr.interpMethod;
px = pr.outPixelSize ;
wienerAlpha = pr.wienerAlpha;
OTFCumThresh = pr.OTFCumThresh;
skewed = pr.skewed;
OTFAreaThresh = pr.OTFAreaThresh;
minIntThrsh = pr.minIntThrsh;

% Check if volume is all zeros
%volume = ts_read_zarr(fn, chunk_i, timepoint_i, channel_i, python_path);
if all(volume == 0, 'all')
    support_ratio.FFTratio_mean = 0;
    support_ratio.FFTratio_median = 0;
    support_ratio.FFTratio_sd = 0;
    support_ratio.embedding_sd = 0;
    support_ratio.OTF_embedding_sum = 0;
    support_ratio.OTF_embedding_vol = 0;
    support_ratio.OTF_embedding_normIntegral = 0;
    
    support_ratio.moment_OTF_embedding_sum = 0;
    support_ratio.moment_OTF_embedding_ideal_sum = 0;
    support_ratio.moment_OTF_embedding_norm = 0;
    support_ratio.integratedPhotons = 0;
    
    return;
end

im = imresize3(single(psf), round(size(psf) .* [xyPixelSize, xyPixelSize, dz] ./ px), 'method', interpMethod);
sz = size(im);
hfN = (N - 1) / 2;
if any(size(im) ~= N)
    if any(sz < N)
        im = padarray(im, max(0, floor((N - sz) / 2)), 0, 'pre');
        im = padarray(im, max(0, ceil((N - sz) / 2)), 0, 'post');
    end
    c = round((size(im) + 1) / 2);
    im = im(c(1) - hfN(1) : c(1) + hfN(1), c(2) - hfN(2) : c(2) + hfN(2), c(3) - hfN(3) : c(3) + hfN(3));
end
psf = im;

%volume = readtiff(fn);
%mode(volume(:))
%volume = volume - mode(volume(:));
%volume(volume(:)<0) = 0;

im = imresize3(single(volume), round(size(volume) .* [xyPixelSize, xyPixelSize, dz] ./ px), 'method', interpMethod);
sz = size(im);
hfN = (N - 1) / 2;
if any(size(im) ~= N)
    if any(sz < N)
        im = padarray(im, max(0, floor((N - sz) / 2)), 0, 'pre');
        im = padarray(im, max(0, ceil((N - sz) / 2)), 0, 'post');
    end
    c = round((size(im) + 1) / 2);
    im = im(c(1) - hfN(1) : c(1) + hfN(1), c(2) - hfN(2) : c(2) + hfN(2), c(3) - hfN(3) : c(3) + hfN(3));
end
volume = im;
if all(volume == 0, 'all')
    support_ratio.FFTratio_mean = 0;
    support_ratio.FFTratio_median = 0;
    support_ratio.FFTratio_sd = 0;
    support_ratio.embedding_sd = 0;
    support_ratio.OTF_embedding_sum = 0;
    support_ratio.OTF_embedding_vol = 0;
    support_ratio.OTF_embedding_normIntegral = 0;
    
    support_ratio.moment_OTF_embedding_sum = 0;
    support_ratio.moment_OTF_embedding_ideal_sum = 0;
    support_ratio.moment_OTF_embedding_norm = 0;
    support_ratio.integratedPhotons = 0;
    
    return;
end

ipvol = imresize3(single(volume), round(size(volume) ./ [xyPixelSize, xyPixelSize, dz] .* px), 'method', interpMethod);
IPhotons = sum(ipvol(:))*.22/.8;

[ abs_OTF_c, OTF_mask] = GU_OTF_FFT_segmentation(psf, wienerAlpha, skewed, 'OTFCumThresh', OTFCumThresh,'OTFAreaThresh', OTFAreaThresh, 'minIntThrsh',minIntThrsh);
[sy,sx,sz] = size(OTF_mask);
D_im = zeros([sy,sx,sz],'logical');
D_im((sy-1)/2,(sx-1)/2,(sz-1)/2) = 1;
D = bwdist(D_im);
perim = bwperim(OTF_mask);
S_psf = D(perim);

[ abs_OTF_c_s, OTF_mask_s] = GU_OTF_FFT_segmentation(volume, wienerAlpha, skewed, 'OTFCumThresh', OTFCumThresh,'OTFAreaThresh', OTFAreaThresh, 'minIntThrsh',minIntThrsh);
D_vol = bwdist(OTF_mask_s);
S_vol = S_psf-D_vol(perim);

OTF_embedding = abs_OTF_c_s./abs_OTF_c;
OTF_embedding_ideal = abs_OTF_c_s./abs_OTF_c;
OTF_embedding_ideal(~OTF_mask) = 0;

OTF_embedding_sum = sum(OTF_embedding(OTF_mask_s));
OTF_embedding_vol = sum(OTF_mask(:));
OTF_embedding_normIntegral = OTF_embedding_sum./OTF_embedding_vol;
OTF_embedding(~OTF_mask_s) = 0;

c = zeros(size(OTF_mask_s), 'logical');
[cy, cx, cz] = size(OTF_mask_s);
c((cy-1)/2, (cx-1)/2, (cz-1)/2) = 1;
c_dist = bwdist(c);
moment_OTF_embedding = c_dist.*OTF_embedding;
moment_OTF_embedding_ideal = c_dist.*OTF_embedding_ideal;

support_ratio.FFTratio_mean = mean(S_vol./S_psf);
support_ratio.FFTratio_median = median(S_vol./S_psf);
support_ratio.FFTratio_sd = std(S_vol./S_psf);
support_ratio.embedding_sd = std(OTF_embedding(OTF_mask_s));
support_ratio.OTF_embedding_sum = OTF_embedding_sum;
support_ratio.OTF_embedding_vol = OTF_embedding_vol;
support_ratio.OTF_embedding_normIntegral = OTF_embedding_normIntegral;

support_ratio.moment_OTF_embedding_sum = sum(moment_OTF_embedding(:));
support_ratio.moment_OTF_embedding_ideal_sum = sum(moment_OTF_embedding_ideal(:));
support_ratio.moment_OTF_embedding_norm = sum(moment_OTF_embedding(:))./sum(moment_OTF_embedding_ideal(:));
support_ratio.integratedPhotons = IPhotons;

return;

end

function create_json_file(fn, chunk_i, timepoint_i, channel_i, support_ratio_avg)
    filename = [fn(1:end-5) '_c' num2str(chunk_i) '_t' num2str(timepoint_i) '_ch' num2str(channel_i) '.json'];
    json_data = containers.Map();
    json_key = [num2str(chunk_i) '.' num2str(timepoint_i) '.0.0.0.' num2str(channel_i)];
    json_data(json_key) = support_ratio_avg;
    modifiedJsonText = jsonencode(json_data);
    fid = fopen(filename, 'w');
    if fid == -1
        error('Cannot open the file for writing: %s', filename);
    end
    fprintf(fid, '%s', modifiedJsonText);
    fclose(fid);
end
