parentDir = '~/Desktop/omg_deep_learning_googlenet/';
dataDir = 'ehza_datasets_COVID';

allImages = imageDatastore(fullfile(parentDir, dataDir),'IncludeSubfolders',true, 'LabelSource', 'foldername');

[imgsTrain, imgsValidation] = splitEachLabel(allImages, 0.80, 'randomized');
disp(['Number of training images: ', num2str(numel(imgsTrain.Files))]);
disp(['Number of validation images: ', num2str(numel(imgsValidation.Files))]);

net = googlenet;

layers = net.Layers;

inputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net); 

numClasses = numel(categories(imgsTrain.Labels));
% nameofClasses = categories(imgsTrain.Labels);

newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);
    
lgraph = replaceLayer(lgraph,'loss3-classifier',newLearnableLayer);

newClassLayer = classificationLayer('Name','with_mask', 'Name', 'without_mask');



lgraph = replaceLayer(lgraph,'output',newClassLayer);


pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimgsTrain = augmentedImageDatastore(inputSize(1:2),imgsTrain, ...
    'DataAugmentation',imageAugmenter);


augimgsValidation = augmentedImageDatastore(inputSize(1:2),imgsValidation);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',100, ...
    'MaxEpochs',1, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimgsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',true, ...
    'ExecutionEnvironment', 'parallel', ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimgsTrain,lgraph,options);



trueLabels = imgsValidation.Labels;
[YPred, probs] = classify(netTransfer, augimgsValidation);
accuracy = mean(YPred == trueLabels);


plotconfusion(trueLabels, YPred);