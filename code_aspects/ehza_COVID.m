parentDir = '~/Desktop/omg_deep_learning/';
dataDir = 'ehza_datasets_COVID';

allImages = imageDatastore(fullfile(parentDir, dataDir),'IncludeSubfolders',true, 'LabelSource', 'foldername');

rng default;

[imgsTrain, imgsValidation] = splitEachLabel(allImages, 0.8, 'randomized');
disp(['Number of training images: ', num2str(numel(imgsTrain.Files))]);
disp(['Number of validation images: ', num2str(numel(imgsValidation.Files))]);

alex = alexnet;

layers = alex.Layers;

layers(end-2) = fullyConnectedLayer(2);
layers(end) = classificationLayer;

inputSize = alex.Layers(1).InputSize;
augimgsTrain = augmentedImageDatastore(inputSize(1:2), imgsTrain);
augimgsValidation = augmentedImageDatastore(inputSize(1:2), imgsValidation);

mbSize = 32;
mxEpochs = 3;
ilr = 1e-3;
plt = 'training-progress';

opts = trainingOptions('sgdm', ...
    'InitialLearnRate', ilr, ...
    'MaxEpochs', mxEpochs, ...
    'MiniBatchSize', mbSize, ...
    'ValidationData', augimgsValidation, ...
    'ValidationFrequency', 75, ...
    'ValidationPatience', 5, ...
    'ExecutionEnvironment', 'parallel', ...
    'plots', plt);
    

trainedAN = trainNetwork(augimgsTrain, layers, opts);