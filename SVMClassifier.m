clc; clear;

%% Section 1 - Load embeddings data
rng('default');
emb = fastTextWordEmbedding;

%% Section 2 - Set up a Matlab table for positive and negative wordlists

loadLexicon; % Run loadLexicon.m script

% Divide opinion words into positive and negative labels
words = [positiveWords; negativeWords];
labels = categorical(nan(numel(words), 1));
labels(1:numel(positiveWords)) = "Positive";   
labels(numel(positiveWords)+1:end) = "Negative";

% Put the reviews and polarity into a table with 2 columns
data = table(words, labels, 'VariableNames', {'Word', 'Label'});

% If the words in the data variable are not contained in the word
% embeddings, remove them from the table
idx = ~isVocabularyWord(emb, data.Word);
data(idx, :) = [];

%% Section 3 - Train an SVM classifier

trainSVM; % Run the trainSVM.m script

% Visualize the classification accuracy in a confusion matrix
figure
confusionchart(YTest, YPred, 'ColumnSummary', 'column-normalized');

%% Section 4 - Applying the trained model on reviews

% Comment/Uncomment to select dataset for analysis
loadRTDataset;
%loadIMDBDataset;

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
    fprintf('Review No.: %d, Text: %s, DetectedScore: %d, TrueScore: %d\n', ...
        i, joinWords(sentences(i)), sentimentScore(i), actualScore(i));
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