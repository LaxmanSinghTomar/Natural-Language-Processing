
# Text data with Scikit Learn

## Word Counts with CountVectorizer


```python
from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]

# create the transforms
vectorizer = CountVectorizer()

# tokenize and build vocab
vectorizer.fit(text)

# summarize
print(vectorizer.vocabulary_)

# encode document
vector = vectorizer.transform(text)

# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())
```

    {'the': 7, 'quick': 6, 'brown': 0, 'fox': 2, 'jumped': 3, 'over': 5, 'lazy': 4, 'dog': 1}
    (1, 8)
    <class 'scipy.sparse.csr.csr_matrix'>
    [[1 1 1 1 1 1 1 2]]
    

### Encoding another document with fit CountVectorizer


```python
text2 = ["the puppy"]
vector = vectorizer.transform(text2)
print(vector.toarray())
```

    [[0 0 0 0 0 0 0 1]]
    

## Word frequencies with Tfidf Vectorizer


```python
from sklearn.feature_extraction.text import TfidfVectorizer

# list of text docs
text = ["The quick brown fox jumped over the lazy dog.",
       "The dog",
       "The fox"]

# create the transforms
vectorizer = TfidfVectorizer()

# tokenize and build vocab
vectorizer.fit(text)

# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)

# encode document
vector = vectorizer.transform([text[0]])

# summarize encoded doc
print(vector.shape)
print(vector.toarray())
```

    {'the': 7, 'quick': 6, 'brown': 0, 'fox': 2, 'jumped': 3, 'over': 5, 'lazy': 4, 'dog': 1}
    [1.69314718 1.28768207 1.28768207 1.69314718 1.69314718 1.69314718
     1.69314718 1.        ]
    (1, 8)
    [[0.36388646 0.27674503 0.27674503 0.36388646 0.36388646 0.36388646
      0.36388646 0.42983441]]
    

## Hashing with HashingVectorizer


```python
from sklearn.feature_extraction.text import HashingVectorizer

# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]

# creating the transform
vectorizer = HashingVectorizer(n_features = 20)

# encoding the doc
vector = vectorizer.transform(text)

#summarize encoded doc
print(vector.shape)
print(vector.toarray())
```

    (1, 20)
    [[ 0.          0.          0.          0.          0.          0.33333333
       0.         -0.33333333  0.33333333  0.          0.          0.33333333
       0.          0.          0.         -0.33333333  0.          0.
      -0.66666667  0.        ]]
    
