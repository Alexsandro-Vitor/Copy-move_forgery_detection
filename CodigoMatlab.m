img = imread('teste 3.bmp')
cinza = rgb2gray(img)

lbpA = extractLBPFeatures(cinza, 'NumNeighbors', 8, 'Radius', 1, 'Upright', true)
lbpB = extractLBPFeatures(cinza, 'NumNeighbors', 12, 'Radius', 2, 'Upright', true)
lbpC = extractLBPFeatures(cinza, 'NumNeighbors', 16, 'Radius', 2, 'Upright', false)

figure
imshow(img)
figure
imshow(cinza)