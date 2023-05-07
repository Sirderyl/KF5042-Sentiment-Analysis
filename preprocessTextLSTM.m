function [sourceText] = preprocessTextLSTM(textData)
%PREPROCESSTEXTLSTM - Used for cleaning and simplyfying the text
sourceText = eraseTags(textData); % Remove HTML tags.
sourceText = tokenizedDocument(sourceText); % Tokenize the text.
sourceText = lower(sourceText); % Convert the text data to lowercase.
sourceText = erasePunctuation(sourceText); % Erase punctuation.
end