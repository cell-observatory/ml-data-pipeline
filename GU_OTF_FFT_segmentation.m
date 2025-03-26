function [abs_OTF_c, OTF_mask] = GU_OTF_FFT_segmentation(psf, skewed, varargin)
% generate OTF masked wiener back projector
%
% Author: Xiongtao Ruan (11/10/2022)
% Gokul: based on XR's omw_backprojector_generation function. This code disables finding the convex hull

ip = inputParser;
ip.CaseSensitive = false;
ip.addRequired('psf'); 
ip.addRequired('skewed'); 
ip.addParameter('OTFCumThresh', 0.9, @isnumeric);
ip.addParameter('OTFAreaThresh', 100, @isnumeric);
ip.addParameter('minIntThrsh', 1e-3, @(x) isnumeric(x));

ip.parse(psf, skewed, varargin{:});

pr = ip.Results;
otf_thresh = pr.OTFCumThresh;
area_thresh = pr.OTFAreaThresh;
minIntThrsh = pr.minIntThrsh;

if isempty(psf) || all(psf == 0, 'all')
    error('A valid nonzero PSF needs to be provided!')
end

psf = single(psf);
psf = psf ./ sum(psf, 'all');

OTF = decon_psf2otf(psf);
abs_OTF = abs(OTF);
abs_OTF_c = fftshift(abs_OTF);
OTF_vals = sort(double(abs_OTF(:)), 'descend');

% segment out OTF support based on the accumulate OTF intensity
tind = find(cumsum(OTF_vals) > sum(OTF_vals) * otf_thresh, 1, 'first');
dc_thresh = OTF_vals(tind) / abs_OTF(1, 1, 1);
% (11/12/2022): add a lower bound 2.5e-3 in case of too noisy
dc_thresh = max(dc_thresh, minIntThrsh);
OTF_mask = abs_OTF_c > abs_OTF(1, 1, 1) * dc_thresh;
OTF_mask = imopen(OTF_mask, strel('sphere', 2));
OTF_mask = bwareaopen(OTF_mask, area_thresh);
OTF_mask = imclose(OTF_mask, strel('sphere', 2));
OTF_mask = imopen(OTF_mask, strel('sphere', 2));
OTF_mask = bwareaopen(OTF_mask, area_thresh);

if ~any(OTF_mask, 'all')
    error('The OTF mask is empty, check the PSF and OTF-related parameters!')
end

% for skewed space data, the OTF mask has three main componets, need to
% concatenate them along z. 
% first automatically decide if the OTF is in skewed space
if isempty(skewed)
    skewed = false;
    OTF_mask_xz = squeeze(sum(OTF_mask, 1)) > 0;
    CC = bwconncomp(OTF_mask_xz);
    % if there are more than one connected components, check if the x projected 
    % line cover the whole z range, and also if peak is at/close to center.
    if CC.NumObjects > 1
        OTF_mask_xy_line = sum(OTF_mask_xz, 1);
        [~, pind] = max(OTF_mask_xy_line);
        if all(OTF_mask_xy_line >0) && abs(pind - (numel(OTF_mask_xy_line) + 1) / 2) > 1
            skewed = true;
        end
    end
end

if skewed
    L = bwlabeln(OTF_mask);
    if numel(unique(L)) > 4
        CC = bwconncomp(OTF_mask, 26);
        vols = cellfun(@numel, CC.PixelIdxList);
        [~, max_inds] = maxk(vols, 3);
        OTF_mask = false(size(OTF_mask));
        OTF_mask(cat(1, CC.PixelIdxList{max_inds})) = true;
        L = bwlabeln(OTF_mask);
    elseif numel(unique(L)) == 2
        L = (L + 1) .* OTF_mask;
    end
    OTF_mask_c = L == 2;
    OTF_mask_l = L == 1;
    OTF_mask_r = L == 3;
    OTF_mask = cat(3, OTF_mask_r, OTF_mask_c, OTF_mask_l);
else
    CC = bwconncomp(OTF_mask, 26);
    [~, max_ind] = max(cellfun(@numel, CC.PixelIdxList));
    OTF_mask = false(size(OTF_mask));
    OTF_mask(CC.PixelIdxList{max_ind}) = true;
end

end
