# Poem Classifier with Markov Model

## Introduction

## Installation
To set up the project environment, follow these steps:
- `import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string

from nltk import WordNetLemmatizer, word_tokenize, pos_tag, SnowballStemmer
from nltk.corpus import stopwords, wordnet

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score`

## Usage
How to run and use the project:
- `# edgar_allan_poe = poem_reader('edgar_allan_poe.txt')
# robert_frost_poe = poem_reader('robert_frost_poe.txt')`
- `# edgar_allan_poe`
- `# robert_frost_poe`
- `edgar_allan_preprocessed = text_preprocessor(edgar_allan_poe['text'])
robert_frost_preprocessed = text_preprocessor(robert_frost_poe['text'])

edgar_allan_preprocessed = edgar_allan_preprocessed.to_frame()
robert_frost_preprocessed = robert_frost_preprocessed.to_frame()`
- `x = all_poems_preprocessed.drop(columns=['label'])
y = all_poems_preprocessed['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

x_train = x_train['text']
x_test = x_test['text']`
- `idx = 1
word2idx = {'<unk>': 0}`
- `x_train_int = []
x_test_int = []

for text in x_train:
    text2idx = []
    tokens = text.split()
    for token in tokens:
        text2idx.append(word2idx[token])
    x_train_int.append(text2idx)

for text in x_test:
    text2idx = []
    tokens = text.split()
    for token in tokens:
        try:
            text2idx.append(word2idx[token])
        except:
            text2idx.append(word2idx['<unk>'])
    x_test_int.append(text2idx)
`
- `V = len(word2idx)

A0 = np.ones((V, V))
PI0 = np.ones(V)

A1 = np.ones((V, V))
PI1 = np.ones(V)`
- `# Get the unique values and their counts
unique_values, counts = np.unique(A1, return_counts=True)

# Combine the unique values and their counts into a dictionary
value_count_dict = dict(zip(unique_values, counts))

# Print the value counts
for value, count in value_count_dict.items():
    print(f"Value {value} occurs {count} times.")`
- `A0 /= A0.sum(axis=1, keepdims=True)
PI0 /= PI0.sum()

A1 /= A1.sum(axis=1, keepdims=True)
PI1 /= PI1.sum()`
- `# Get the unique values and their counts
unique_values, counts = np.unique(A1, return_counts=True)

# Combine the unique values and their counts into a dictionary
value_count_dict = dict(zip(unique_values, counts))

# Print the value counts
for value, count in value_count_dict.items():
    print(f"Value {value} occurs {count} times.")`
- `# Log A and PI Since We Don't Need The Actual Probs
logA0 = np.log(A0)
logPI0 = np.log(PI0)

logA1 = np.log(A1)
logPI1 = np.log(PI1)`
- `clf = Classifier([logA0, logA1], [logPI0, logPI1], [logp0, logp1])`
- `Ptrain = clf.predict(x_train_int)
print(f'Train acc: {np.mean(Ptrain == y_train)}')`
- `Ptest = clf.predict(x_test_int)
print(f'Test acc: {np.mean(Ptest == y_test)}')`

## Code Description
A brief overview of the main components of the code:
```python
# def poem_reader(input_file_path):
#     
#     try:
#         with open(input_file_path, 'r') as input_file:
#             content = input_file.read()  # Read the content of the input text file
#         
#         sections = content.split('\n\n')  # Split content into sections based on empty lines
#         data = {'text': sections}        
#         print('Done')
#         
#         return pd.DataFrame(data)
#         
#     except FileNotFoundError:
#         print("Input file not found.")
#     except Exception as e:
#         print("An error occurred:", e)
```
```python
def text_preprocessor(data):
    # Load English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Initialize the SnowballStemmer with English language
    # stemmer = SnowballStemmer("english")
    
    # Initialize the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Mapping between Penn Treebank POS tags and WordNet POS tags
    def penn_to_wordnet_pos(tag):
        if tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to noun
    
    # Tokenize and preprocess the text data
    def preprocess_text(text):
        tokens = word_tokenize(text)  # Tokenize the text
        tokens = [token.lower() for token in tokens if token.isalpha()]  # Keep only alphabetic tokens
        tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
        
        # stemmed_tokens = [stemmer.stem(token) for token in tokens]  # Apply Snowball stemmer
        # return " ".join(stemmed_tokens)  # Join the tokens back into a string
        
        pos_tags = pos_tag(tokens)  # Get part of speech tags
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token, pos in pos_tags]  # Apply lemmatization without POS
        lemmatized_tokens = [lemmatizer.lemmatize(token, pos=penn_to_wordnet_pos(pos)) for token, pos in pos_tags]  # Apply lemmatization with POS
        return " ".join(lemmatized_tokens)  # Join the tokens back into a string
    
    return data.apply(preprocess_text)
```
```python
def compute_counts (text_as_int, A, PI):
    for tokens in text_as_int:
        last_idx = None
        for idx in tokens:
            if last_idx is None:
                # it's the first word in the sentence
                PI[idx] += 1
            else:
                # The last word exist, so count a transition
                A[last_idx, idx] += 1
            
            # Update idx
            last_idx = idx


compute_counts([t for t, y in zip(x_train_int, y_train) if y == 0], A0, PI0)
compute_counts([t for t, y in zip(x_train_int, y_train) if y == 1], A1, PI1)
```
```python
class Classifier:
    def __init__(self, logAs, logpis, logpriors):
        self.logAs = logAs
        self.logpis = logpis
        self.logpriors = logpriors
        self.K = len(logpriors) # number of classes
    
    def _compute_log_likelihood(self, input_, class_):
        logA = self.logAs[class_]
        logpi = self.logpis[class_]
        
        last_idx = None
        logprob = 0
        
        for idx in input_:
            if last_idx is None:
                # it's the first token
                logprob += logpi[idx]
            else:
                logprob += logA[last_idx, idx]
            
            #update last_idx
            last_idx = idx
        
        return logprob
    
    def predict(self, inputs):
        predictions = np.zeros(len(inputs))
        
        for i, input_ in enumerate(inputs):
            posteriors = [self._compute_log_likelihood(input_, c) + self.logpriors[c]  for c in range(self.K)]
            pred = np.argmax(posteriors)
            predictions[i] = pred
        
        return predictions

```

## Results
The main findings or outputs of the project:
```python
# Read the text file and store its content in a list
with open('edgar_allan_poe.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Create a dataframe with one column 'Text' and populate it with the lines from the text file
edgar_allan_poe= pd.DataFrame({'text': lines})

edgar_allan_poe
```
```python
# Read the text file and store its content in a list
with open('robert_frost_poe.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Create a dataframe with one column 'Text' and populate it with the lines from the text file
robert_frost_poe = pd.DataFrame({'text': lines})

robert_frost_poe
```
```python
nltk.download('punkt')  # Download NLTK data for tokenization if not already downloaded
nltk.download('stopwords')  # Download NLTK data for stopwords if not already downloaded
nltk.download('wordnet')  # Download NLTK data for WordNet if not already downloaded
nltk.download('averaged_perceptron_tagger')  # Download NLTK data for POS tagger if not already downloaded
```
```python
edgar_allan_preprocessed
```
```python
robert_frost_preprocessed
```
```python
edgar_allan_preprocessed['label'] = 0
robert_frost_preprocessed['label'] = 1

edgar_allan_preprocessed
```
```python
# Remove rows with empty cells in the 'Text' column
edgar_allan_preprocessed['text'].replace('', pd.NA, inplace=True)
edgar_allan_preprocessed.dropna(subset=['text'], inplace=True)

# Reset the index after dropping rows
edgar_allan_preprocessed.reset_index(drop=True, inplace=True)

edgar_allan_preprocessed
```
```python
# Remove rows with empty cells in the 'Text' column
robert_frost_preprocessed['text'].replace('', pd.NA, inplace=True)
robert_frost_preprocessed.dropna(subset=['text'], inplace=True)

# Reset the index after dropping rows
robert_frost_preprocessed.reset_index(drop=True, inplace=True)

robert_frost_preprocessed
```
```python
all_poems_preprocessed = pd.concat([edgar_allan_preprocessed, robert_frost_preprocessed], ignore_index=True)

all_poems_preprocessed
```
```python
x_train
```
```python
y_train
```
```python
for text in x_train:
    tokens = text.split()
    for token in tokens:
        if token not in word2idx:
            word2idx[token] = idx
            idx += 1

word2idx
```
```python
x_train_int
```
```python
x_test_int
```
```python
PI1
```
```python
A1
```
```python
A1
```
```python
PI1
```
```python
count0 = sum(y == 0 for y in y_train)
count1 = sum(y == 1 for y in y_train)
total = len(y_train)
p0 = count0 / total
p1 = count1 / total
logp0 = np.log(p0)
logp1 = np.log(p1)

p0, p1
```
```python
cm = confusion_matrix(y_train, Ptrain)

cm
```
```python
cm_test = confusion_matrix(y_test, Ptest)

cm_test
```
```python
f1_score(y_train, Ptrain)
```
```python
f1_score(y_test, Ptest)
```

## Conclusion

