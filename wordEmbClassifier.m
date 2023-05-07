%% Section 1 - Load embeddings data
rng('default');
emb = fastTextWordEmbedding;

%% Section 2 - Set up a Matlab table for positive and negative wordlists
words = [positiveWords; negativeWords];
labels = categorical(nan(numel(words), 1));
labels(1:numel(positiveWords)) = "Positive";   
labels(numel(positiveWords)+1:end) = "Negative";

data = table(words, labels, 'VariableNames', {'Word', 'Label'});

% If the words in the data variable are not contained in the word
% embeddings, remove them from the dataset
idx = ~isVocabularyWord(emb, data.Word);
data(idx, :) = [];

%% Section 3 - Train an SVM classifier

% Split data train/test
numWords = size(data, 1);
cvp = cvpartition(numWords, 'HoldOut', 0.01);
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

% Visualize the classification accuracy in a confusion matrix
figure
confusionchart(YTest, YPred, 'ColumnSummary', 'column-normalized');

%% Section 4 - Applying the trained model on reviews

% Removing any words that are not in the word embeddings
idx = ~isVocabularyWord(emb, sentences.Vocabulary);
removeWords(sentences, idx);

sentimentScore = zeros(size(sentences));

% For evaluation
tp = 0;
tn = 0;

% Loop over the sentences and sum up the sentiment bearing words used
% zero is returned if there are no sentiment words or there are balanced
% positive and negative
for i = 1 : sentences.length
    docwords = sentences(i).Vocabulary;
    vec = word2vec(emb, docwords);
    [~, scores] = predict(model, vec);
    sentimentScore(i) = mean(scores(:, 1));
    % Replace NaN values with zeros for accuracy comparisons
    if isnan(sentimentScore(i))
        sentimentScore(i) = 0;
    end
    fprintf('Sent: %d, words: %s, FoundScore: %d, GoldScore: %d\n', i, ...
        joinWords(sentences(i)), sentimentScore(i), actualScore(i));
    if sentimentScore(i) > 0 && actualScore(i) == 1
        tp = tp + 1;
    end
    if sentimentScore(i) < 0 && actualScore(i) == 0
        tn = tn + 1;
    end
end

% Quantitative Evaluation of Sentiment Analysis
missed = sum(sentimentScore == 0);
found = numel(sentimentScore) - missed;
coverage = found / numel(sentimentScore) * 100
accuracy = (tp + tn) * 100 / found