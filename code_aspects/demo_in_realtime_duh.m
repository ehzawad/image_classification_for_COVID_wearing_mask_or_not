clear all
cam = webcam(1);


myCNN = matfile('~/Desktop/omg_deep_learning/TrainedModel/ehza_Alex_for_COVID_mask.mat');

clc
for frames = 1:100
    img = snapshot(cam);
    img = imresize(img, [227, 227]);
    [label, score] = classify(myCNN.trainedAN, img);
    disp(label);
    imshow(img);
    title({char(label), num2str(max(score), 2)});
    
end