clc; clear;

rng('default');
emb = fastTextWordEmbedding;

%% Section 1 - Load the opinion lexicon

loadLexicon; % Run the loadLexicon.m script

% Define words_hash as a content addressable store (dictionary)
words_hash = java.util.Hashtable;

% Putting the positive words in the hashtable, giving them a value of 1
[possize, ~] = size(positiveWords);
for i = 1 : possize
    words_hash.put(positiveWords(i, 1), 1);
end

% The same for negative, just with a value of -1
[negsize, ~] = size(negativeWords);
for i = 1 : negsize
    words_hash.put(negativeWords(i, 1), -1);
end

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

%% Section 2 - Train the SVM classifier

trainSVM; % Run the trainSVM.m script

%% Section 3 - Load the dataset

% Comment/Uncomment to select dataset to analyze
loadRTDataset;
%loadIMDBDataset;

%% Section 4 - Evaluating the reviews

sentimentScore = zeros(size(sentences)); % vector containing sentiment results.

% Initializing true positive and true negative variables for evaluation
tp = 0;
tn = 0;

% Loop over the sentences and sum up the sentiment bearing words used
% zero is returned if there are no sentiment words or there are balanced
% positive and negative
for i = 1 : sentences.length
    docwords = sentences(i).Vocabulary;
    for j = 1 : length(docwords)
        if words_hash.containsKey(docwords(j))
            sentimentScore(i) = sentimentScore(i) + words_hash.get(docwords(j));
        end
    end
    if sentimentScore(i) == 0
        vec = word2vec(emb, docwords);
        [~, scores] = predict(model, vec);
        sentimentScore(i) = mean(scores(:, 1));
    end
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