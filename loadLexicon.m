filePositive = fopen(fullfile('opinion-lexicon-English', 'positive-words.txt'));
fileNegative = fopen(fullfile('opinion-lexicon-English', 'negative-words.txt'));

% Skip comment lines
posScan = textscan(filePositive, '%s', 'CommentStyle', ';');
negScan = textscan(fileNegative, '%s', 'CommentStyle', ';');

positiveWords = string(posScan{1});
negativeWords = string(negScan{1});
fclose all;