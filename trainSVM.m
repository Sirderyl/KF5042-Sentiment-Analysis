% Split data train/test - 90%/10%
numWords = size(data, 1);
cvp = cvpartition(numWords, 'HoldOut', 0.1);
dataTrain = data(training(cvp), :);
dataTest = data(test(cvp), :);

% Using word2vec to convert words into vectors
wordsTrain = dataTrain.Word;
XTrain = word2vec(emb, wordsTrain);
YTrain = dataTrain.Label;

% Train an SVM sentiment classifier to classify vectors into positive and
% negative categories
model = fitcsvm(XTrain, YTrain);

% Test classifier
wordsTest = dataTest.Word;
XTest = word2vec(emb, wordsTest);
YTest = dataTest.Label;

% Predict sentiments on test words
[YPred, scores] = predict(model, XTest);