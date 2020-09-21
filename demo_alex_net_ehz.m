img1 = imread('~/Desktop/faces.jpg');
img_resized_for_deepnet = imresize(img1, [227 227]);
pred = classify(trainedAN, img_resized_for_deepnet);