#
# I wanted to created a predicted output csv to mimic the sampleSubmission csv we were given.  Using a Functional
# model I ran into some issues getting it to predict the classes, but I worked that out.  Once I figured that out
# I was able to predict the test tsv and output a predicted csv with corresponding PhraseId.
#

import tensorflow as tf
import pandas as pd
import csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# call of saved model
model1 = tf.keras.models.load_model('model.h1')

# IMPORT TEST DATA
data = pd.read_csv('test.tsv', delimiter='\t')

tokenizer = Tokenizer(num_words=100, lower=True, char_level=False)
tokenizer.fit_on_texts(data['Phrase'].tolist())
testing_sequences = tokenizer.texts_to_sequences(data["Phrase"].tolist())
test_word_index = tokenizer.word_index  # NOT NEEDED?
test_cnn_data = pad_sequences(testing_sequences, maxlen=100)
length = (len(data['Phrase'])) - 2

# GET PREDICTION, SINCE DIDN'T USE SEQUENTIAL, HAD TO FIND WAY TO PREDICT CLASSES
predictions = model1.predict(test_cnn_data)
classes = predictions.argmax(axis=1)

#THESE TO CHECK OUTPUTS
print(data['PhraseId'][0])
print(data['Phrase'][0])
print(classes[0], '\n')

# CREATE NEW CSV FILE TO OUTPUT TEST
with open('Exam_2.1_Output.csv', 'w', newline='') as csvfile:
    fieldnames = ['PhraseID', 'Sentiment']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # PREDICTS EVERY LINE OF TEST CSV AND PLACES PHRASE ID
    for index in range(length):
        writer.writerow({'PhraseID': data['PhraseId'][index], 'Sentiment': classes[index]})

csvfile.close()

# REOPENED TO USE VALUE_COUNTS
info = pd.read_csv('Exam_2.1_Output.csv')

# CHECK SENTIMENT TOTALS PER CLASS
print(info['Sentiment'].value_counts(), '\n')
