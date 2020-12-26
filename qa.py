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