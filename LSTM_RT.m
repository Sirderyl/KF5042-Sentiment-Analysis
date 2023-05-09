clc; clear;

%% Section 1 - Importing dataset

filename = "rt_dataset.csv";
reviewData = readtable(filename, 'TextType', 'string');

% Convert sentiment polarity to categorical data to fit LSTM
reviewData.score = categorical(reviewData.score);

%% Section 2 - Partitioning

% Split 80% training 20% testing
cvp = cvpartition(reviewData.score, 'HoldOut', 0.2);
dataTrain = reviewData(training(cvp), :);
dataTest = reviewData(test(cvp), :);

%sentTrain and sentTest will be further processed to XTrain and XTest
sentTrain = dataTrain.review;
sentTest = dataTest.review;
YTrain = dataTrain.score;
YTest = dataTest.score;

%% Section 3 - Preprocess reviews

sentencesTrain = preprocessTextLSTM(sentTrain);
sentencesTest = preprocessTextLSTM(sentTest);

%% Section 4 - Convert to sequences

enc = wordEncoding(sentencesTrain);

% Histogram for sentence length determination

% sentencesLength = doclength(sentencesTrain);
% figure
% histogram(sentencesLength)
% title("Sent lengths")
% xlabel("Length")
% ylabel("Number of sentences")

sentLength = 35; % Truncate sentences to same length without losing much data
XTrain = doc2sequence(enc, sentencesTrain, 'Length', sentLength);
XTest = doc2sequence(enc, sentencesTest, 'Length', sentLength);

%% Section 5 - Train LSTM network

inputSize = 1;
dimensions = 50;
hiddenUnits = 80;

numWords = enc.NumWords;
numClasses = numel(categories(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(dimensions, numWords)
    lstmLayer(hiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MiniBatchSize', 16, ...
    'GradientThreshold', 2, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XTest, YTest}, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ValidationPatience', 10);

net = trainNetwork(XTrain, YTrain, layers, options);

%% Section 6 - accuracy calculation

YPred = classify(net, XTest);
acc = sum(YPred == YTest)./numel(YTest);