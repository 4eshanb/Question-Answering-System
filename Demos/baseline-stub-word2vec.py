#!/usr/bin/env python
'''
Created on May 14, 2014
@author: reid

Modified on Nov 6, 2019
Modified on Nov 14, 2019
'''

import sys, nltk, operator
from qa_engine.base import QABase
from sklearn.metrics.pairwise import cosine_similarity
from word2vec_extractor import Word2vecExtractor


# The standard NLTK pipeline for POS tagging a document
def get_sentences(data,type_="story"):
    if type_ == "story":
        sentences = []
        for sent_data in data:
            sentences += [sent_data["sentence"]]
    elif type_ == "question":
        sentences = nltk.sent_tokenize(data)
    else:
        print("ERROR: bad type")
        return []
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    
    return sentences	

def get_bow(tagged_tokens, stopwords):
    return set([t[0].lower() for t in tagged_tokens if t[0].lower() not in stopwords])

def find_phrase(tagged_tokens, qbow):
    for i in range(len(tagged_tokens) - 1, 0, -1):
        word = (tagged_tokens[i])[0]
        if word in qbow:
            return tagged_tokens[i+1:]


# qtokens: is a list of pos tagged question tokens with SW removed
# sentences: is a list of pos tagged story sentences
# stopwords is a set of stopwords
def baseline(qbow, sentences, stopwords):
    # Collect all the candidate answers
    answers = []
    for sent in sentences:
        # A list of all the word tokens in the sentence
        sbow = get_bow(sent, stopwords)
        
        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(qbow & sbow)
        
        answers.append((overlap, sent))
        
    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    # Return the best answer
    best_answer = (answers[0])[1]    
    return best_answer


def baseline_word2vec(question, sentences, stopwords, W2vecextractor):
    q_feat = W2vecextractor.sen2vec(question['question'])
    candidate_answers = []

    for sent in sentences:
        s_feat = W2vecextractor.sen2vec(sent['sentence'])
        dist = cosine_similarity([q_feat], [s_feat])[0][0]
        candidate_answers.append((dist, sent))
        # print("distance: "+str(dist)+"\t sent: "+sent)

    answers = sorted(candidate_answers, key=operator.itemgetter(0), reverse=True)
    best_answer = (answers[0])[1]
    return best_answer



if __name__ == '__main__':

    question_id = "fables-01_Q1"

    driver = QABase()
    q = driver.get_question(question_id)
    story = driver.get_story(q["storyid"])
    question = q["question"]
    print("question:", question)
    stopwords = set(nltk.corpus.stopwords.words("english"))

    qbow = get_bow(get_sentences(question,type_="question")[0], stopwords)
    sentences = get_sentences(story,type_="story")

    # Demo of BOW
    answer = baseline(qbow, sentences, stopwords)
    print("BOW answer:", " ".join(t[0] for t in answer))

    # Demo of word2vec
    glove_w2v_file = "data/glove-w2v.txt"
    W2vecextractor = Word2vecExtractor(glove_w2v_file)

    answer = baseline_word2vec(q, story, stopwords, W2vecextractor)

    print("Word2Vec answer: " + str(answer['sentence']))
