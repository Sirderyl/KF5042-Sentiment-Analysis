filename = "rt_dataset.csv";

dataReviews = readtable(filename, 'TextType', 'string');
textData = dataReviews.review; % get review text
actualScore = dataReviews.score; % get review sentiment

sentences = preprocessText(textData);