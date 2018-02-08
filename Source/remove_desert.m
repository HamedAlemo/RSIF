function B = remove_desert(B, mask)

[~, ~, k] = size(B.MODISReflectance);
mask3D = repmat(mask, 1, 1, k);
B.MODISReflectance(mask3D==1) = NaN;