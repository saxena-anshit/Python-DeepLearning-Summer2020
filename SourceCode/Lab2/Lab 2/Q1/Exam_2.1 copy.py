# a. Include Embedding layer in the design of your models and report if that leads to a better performance
#
# In my design for the model, I initially had a sequential model without embedding, with embedding the model ran
# better by gaining a slightly better performance.  I then switched to a Functional model with embedding and received
# an even better performance.
#
# b. Plot loss of the model and report if you see any overfitting problem
#
# Overfitting was more of an issue when I had the sequential model with embedding.  When I switched to the Functional
# model I still had issues with overfitting, but it was a little better.
#
# c. What techniques you can apply to fix overfitting model
#
# I applied some dropout layers to help with overfitting.  The problem is no where near as bad, however, I could have
# probably had a higher dropout.
#

import numpy as np
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, concatenate
from keras import Input, Model
from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from nltk import word_tokenize
from gensim.models import Word2Vec
import time

# possibly remove punctuation and capitalization

start_time = time.time()

# READ IN TSV FILES
data = pd.read_csv('train.tsv', delimiter='\t')

print(data.shape, '\n')  # Check shape

print(data['Sentiment'].value_counts(), '\n')  # Look at sentiment values

data['Phrase'] = data['Phrase'].apply(word_tokenize)  # Tokenize phrase column

print(data['Phrase'].head(5), '\n')  # Check output

# split data
data_train, data_test = train_test_split(data, test_size=0.10, random_state=42)

# TRAINING DATA INFO
all_training_words = [word for tokens in data_train["Phrase"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in data_train["Phrase"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
print("Max sentence length is %s" % max(training_sentence_lengths), '\n')

# TEST DATA INFO
all_test_words = [word for tokens in data_test['Phrase'] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in data_test['Phrase']]
TEST_VOCAB = sorted(list(set(all_test_words)))
print('%s words total, with a vocabulary size of %s' % (len(all_test_words), len(TEST_VOCAB)))
print('Max sentence length is %s' % max(test_sentence_lengths), '\n')

# TOKENIZE BOTH DATA SETS
tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(data_train['Phrase'].tolist())
tokenizer.fit_on_texts(data_test['Phrase'].tolist())
training_sequences = tokenizer.texts_to_sequences(data_train["Phrase"].tolist())
testing_sequences = tokenizer.texts_to_sequences(data_test["Phrase"].tolist())
train_word_index = tokenizer.word_index
test_word_index = tokenizer.word_index  # NOT NEEDED?
print('Found %s unique tokens.' % len(train_word_index))
train_cnn_data = pad_sequences(training_sequences, maxlen=100)  # GUESSED ON MAXLEN
test_cnn_data = pad_sequences(testing_sequences, maxlen=100)

# NOT TOO SURE ABOUT THESE, BUT RESULTS SEEM OKAY
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 100

# SETUP WORD2VEC
word2vec = Word2Vec(data['Phrase'], min_count=1, size=100, workers=3, window=3, sg=1)

# ESTABLISH WEIGHTS FOR TRAINING
train_embedding_weights = np.zeros((len(train_word_index) + 1, EMBEDDING_DIM))
for word, index in train_word_index.items():
    train_embedding_weights[index, :] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)


# SETUP FOR CNN -> USED KERAS FUNCTIONAL API INSTEAD OF SEQUENTIAL
# IT IS THE ADDITION OF CONVOLUTION LAYERS THAT IMPLEMENT A CNN
def CNN(embeddings, max_sequence_length, num_words, embedding_dim, sentiment_index):
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embeddings],
                                input_length=max_sequence_length,
                                trainable=False)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    convs = []
    filter_sizes = [2, 3, 4, 5, 6]
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=200,
                        kernel_size=filter_size,
                        activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)
    l_merge = concatenate(convs, axis=1)
    x = Dropout(0.1)(l_merge)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    preds = Dense(sentiment_index, activation='softmax')(x)
    model = Model(inputs=sequence_input, outputs=preds)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model


# ESTABLISH CNN MODEL
model = CNN(train_embedding_weights,
            MAX_SEQUENCE_LENGTH,
            len(train_word_index) + 1,
            EMBEDDING_DIM,
            5)


# TRAINING SETUP
x_train = train_cnn_data
y_train = data_train['Sentiment']
x_test = test_cnn_data
y_test = data_test['Sentiment']
num_epochs = 10
batch_size = 32

# HISTORY FOR LATER USE
history = model.fit(x_train,
                    y_train,
                    epochs=num_epochs,
                    validation_split=0.1,
                    shuffle=True,
                    batch_size=batch_size)

[test_loss, test_acc] = model.evaluate(x_test, y_test)

print(history.history.keys())

# SAVE MODEL FOR PREDICTIONS
model.save('./model' + '.h1')

# OUTPUT TIME TO RUN BECAUSE I WAS CURIOUS
print("Execution time --- %s seconds ---" % (time.time() - start_time))

# SUMMARIZE HISTORY FOR ACCURACY
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# SUMMARIZE HISTORY FOR LOSS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
