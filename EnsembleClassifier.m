%% Section 1 - Load the dataset

%filename = "rt_dataset.csv";
filename = "IMDB_Dataset.csv";

dataReviews = readtable(filename, 'TextType', 'string');
textData = dataReviews.review; % get review text
actualScore = dataReviews.score; % get review sentiment

sentences = preprocessText(textData);
fprintf('File: %s, Sentences: %d \n', filename, size(sentences));

%% Section 2 - Evaluating the reviews

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