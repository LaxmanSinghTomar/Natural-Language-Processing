
# How to Prepare Text Data with Keras

## Split words with text_to_word_sequence


```python
from keras.preprocessing.text import text_to_word_sequence

# define the doc
text = 'The quick brown fox jumps over the lazy dog'

# tokenize the document
result = text_to_word_sequence(text)
print(result)
```

    Using TensorFlow backend.
    

    ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
    

## Encoding with One-hot


```python
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot

# define the doc
text = 'The quick brown fox jumps over the lazy dog.'

#estimate the size of the vocab
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)

# integer encode the document
result = one_hot(text, round(vocab_size*1.3))
print(result)
```

    8
    [3, 9, 3, 8, 1, 1, 3, 4, 9]
    

## Hashing Encoding with hashing-trick


```python
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import hashing_trick

# define the doc
text = 'The quick brown fox jumps over the lazy dog.'

#estimate the size of the vocab
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)

# integer encode the document
result = hashing_trick(text, round(vocab_size*1.3), hash_function = 'md5')
print(result)
```

    8
    [6, 4, 1, 2, 7, 5, 6, 2, 6]
    

## Tokenizer API


```python
from keras.preprocessing.text import Tokenizer

# define 5 docs
docs = ['Well done!',
       'Good work',
       'Great effort',
       'nice work',
       'Excellent!']

# create the tokenizer
t = Tokenizer()

# fit the tokenizer on the docs
t.fit_on_texts(docs)

# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)

# integer encode docs
encoded_docs = t.texts_to_matrix(docs, mode = 'count')
print(encoded_docs)
```

    OrderedDict([('well', 1), ('done', 1), ('good', 1), ('work', 2), ('great', 1), ('effort', 1), ('nice', 1), ('excellent', 1)])
    5
    {'work': 1, 'well': 2, 'done': 3, 'good': 4, 'great': 5, 'effort': 6, 'nice': 7, 'excellent': 8}
    defaultdict(<class 'int'>, {'done': 1, 'well': 1, 'work': 2, 'good': 1, 'effort': 1, 'great': 1, 'nice': 1, 'excellent': 1})
    [[0. 0. 1. 1. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 1. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1.]]
    
