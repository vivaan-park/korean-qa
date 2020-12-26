from functools import reduce

from nltk import FreqDist
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

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