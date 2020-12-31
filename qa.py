# © 2020 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from functools import reduce
import os
import argparse

from nltk import FreqDist
import numpy as np
from ckonlpy.tag import Twitter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Permute, dot, add, concatenate
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Activation

def read_data(dir):
    stories, questions, answers = [], [], []
    story_temp = []
    lines = open(dir, 'rb')

    for line in lines:
        line = line.decode('utf-8')
        line = line.strip()
        idx, text = line.split(' ', 1)

        if int(idx) == 1:
            story_temp = []

        if '\t' in text:
            question, answer, _ = text.split('\t')
            stories.append([i for i in story_temp if i])
            questions.append(question)
            answers.append(answer)
        else:
            story_temp.append(text)

    lines.close()
    return stories, questions, answers

def tokenize(twitter, sent):
    return twitter.morphs(sent)

def preprocess_data(twitter, train_data, test_data):
    counter = FreqDist()
    flatten = lambda data: reduce(lambda x, y: x + y, data)

    story_len = []
    question_len = []

    for stories, questions, answers in [train_data, test_data]:
        for story in stories:
            stories = tokenize(twitter, flatten(story))
            story_len.append(len(stories))
            for word in stories:
                counter[word] += 1
        for question in questions:
            question = tokenize(twitter, question)
            question_len.append(len(question))
            for word in question:
                counter[word] += 1
        for answer in answers:
            answer = tokenize(twitter, answer)
            for word in answer:
                counter[word] += 1

    word2idx = {word : (idx + 1) for idx, (word, _) in enumerate(counter.most_common())}
    idx2word = {idx : word for word, idx in word2idx.items()}

    story_max_len = np.max(story_len)
    question_max_len = np.max(question_len)

    return word2idx, idx2word, story_max_len, question_max_len

def vectorize(twitter, data, word2idx, story_maxlen, question_maxlen):
    Xs, Xq, Y = [], [], []
    flatten = lambda data: reduce(lambda x, y: x + y, data)

    stories, questions, answers = data
    for story, question, answer in zip(stories, questions, answers):
        xs = [word2idx[i] for i in tokenize(twitter, flatten(story))]
        xq = [word2idx[i] for i in tokenize(twitter, question)]
        Xs.append(xs)
        Xq.append(xq)
        Y.append(word2idx[answer])

    return pad_sequences(Xs, maxlen=story_maxlen), \
           pad_sequences(Xq, maxlen=question_maxlen), \
           to_categorical(Y, num_classes=len(word2idx) + 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--test-file', required=True)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--embed-size', type=int, default=50)
    parser.add_argument('--lstm-size', type=int, default=64)
    parser.add_argument('--dropout-rate', type=int, default=0.3)

    args = parser.parse_args()
    train_file = os.path.join(args.train_file)
    test_file = os.path.join(args.test_file)
    epochs = args.epochs
    batch_size = args.batch_size
    embed_size = args.embed_size
    lstm_size = args.lstm_size
    dropout_rate = args.dropout_rate

    train_data = read_data(train_file)
    test_data = read_data(test_file)

    twitter = Twitter()
    twitter.add_dictionary('은경이', 'Noun')
    twitter.add_dictionary('경임이', 'Noun')
    twitter.add_dictionary('수종이', 'Noun')

    word2idx, idx2word, story_max_len, question_max_len = \
        preprocess_data(twitter, train_data, test_data)

    vocab_size = len(word2idx) + 1

    Xstrain, Xqtrain, Ytrain = vectorize(
        twitter,
        train_data,
        word2idx,
        story_max_len,
        question_max_len
    )
    Xstest, Xqtest, Ytest = vectorize(
        twitter,
        test_data,
        word2idx,
        story_max_len,
        question_max_len
    )

    input_sequence = Input((story_max_len,))
    question = Input((question_max_len,))

    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(
        input_dim=vocab_size,
        output_dim=embed_size)
    )
    input_encoder_m.add(Dropout(dropout_rate))

    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(
        input_dim=vocab_size,
        output_dim=question_max_len)
    )
    input_encoder_c.add(Dropout(dropout_rate))

    question_encoder = Sequential()
    question_encoder.add(Embedding(
        input_dim=vocab_size,
        output_dim=embed_size,
        input_length=question_max_len)
    )
    question_encoder.add(Dropout(dropout_rate))

    input_encoded_m = input_encoder_m(input_sequence)
    input_encoded_c = input_encoder_c(input_sequence)
    question_encoded = question_encoder(question)

    match = dot([input_encoded_m, question_encoded], axes=-1, normalize=False)
    match = Activation('softmax')(match)

    response = add([match, input_encoded_c])
    response = Permute((2, 1))(response)

    answer = concatenate([response, question_encoded])
    answer = LSTM(lstm_size)(answer)
    answer = Dropout(dropout_rate)(answer)
    answer = Dense(vocab_size)(answer)
    answer = Activation('softmax')(answer)

    model = Model([input_sequence, question], answer)
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    model.fit(
        [Xstrain, Xqtrain],
        Ytrain,
        batch_size,
        epochs,
        validation_data=([Xstest, Xqtest], Ytest)
    )

    model.save('model.h5')

    ytest = np.argmax(Ytest, axis=1)
    Ytest_ = model.predict([Xstest, Xqtest])
    ytest_ = np.argmax(Ytest_, axis=1)

    NUM_DISPLAY = 30

    print('질문                       |실제값     |예측값')
    print('-' * 46)

    for i in range(NUM_DISPLAY):
        question = ' '.join([idx2word[j] for j in Xqtest[i].tolist()])
        label = idx2word[ytest[i]]
        prediction = idx2word[ytest_[i]]
        if len(label) == 2:
            print(f'{question:20}: {label:8} {prediction}')
        else:
            print(f'{question:20}: {label:7} {prediction}')

if __name__ == '__main__':
    main()