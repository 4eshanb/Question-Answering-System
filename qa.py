from qa_engine.base import QABase
import pandas as pd
from tabulate import tabulate
import nltk
import operator
import os
from prettytable import PrettyTable
from word2vec_extractor import Word2vecExtractor
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
#from earthy.nltk_wrappers import porter_stem
from nltk.stem.porter import *
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from gensim.models import Word2Vec
from nltk.stem import SnowballStemmer
import re
import string
from nltk.util import ngrams
import gensim
from pywsd import disambiguate
from pywsd.similarity import max_similarity as maxsim
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import spacy
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deeppavlov import build_model, configs

glove_w2v_file = "data/glove-w2v.txt"
w2vecmodel = gensim.models.KeyedVectors.load_word2vec_format(glove_w2v_file)
model_qa = build_model(configs.squad.squad_bert, download=True)

def get_answer(question, story):
    """
    :param question: dict
    :param story: dict
    :return: sentenceid, answer

    question is a dictionary with keys:
        question -- The raw text of question.
        storyid --  The story id.
        questionid  --  The id of the question.
        ner -- Stanford CoreNLP version of NER
        const_parse -- Stanford CoreNLP version of constituency parse
        dep_parse -- Stanford CoreNLP version of dependency parse


    story is a list of sentence, each sentence is a dictionary with keys:
        storytitle -- the title of the story.
        storyid --  the id of the story.
        sentence -- the raw text sentence version.
        sentenceid --  the id of the sentence
        ner -- Stanford CoreNLP version of NER
        coref -- Stanford CoreNLP version of coreference resolution of the entire story
        const_parse -- Stanford CoreNLP version of constituency parse
        dep_parse -- Stanford CoreNLP version of dependency parse


    """
    ###     Your Code Goes Here         ###

    q = question["question"]
    storyid = question["storyid"]
    questionid = question["questionid"]
    story_id_sent = [(dic["sentenceid"], nltk.sent_tokenize(dic["sentence"])) for dic in story ]

    #answerid = get_synset_sum_overlap_test(q,story_id_sent) #63.21

    #sent_id = lemmatized_pos_selected_overlap(question, story)
    sent_id = bert(q, story_id_sent, story)
    
    first_q_word = get_question_word(q)

    answer = '-'
    
    # process who, why, when, where, which, did

    if sent_id != '-' :
        for dic in story:
            answer_sent_id = dic['sentenceid']
            if answer_sent_id == sent_id:
                answer = str(get_answer_parse_tree(answer_sent_id, dic['sentence'], q, dic['const_parse'], dic['dep_parse'], first_q_word, question['dep_parse'], dic["coref"]))
                break
    else:
        answer = '-'


    if answer == "":
        answer = "-"


    #print(q)
    #print(answer)


    #print("ANSWER", answer)
    #raise NotImplemented 
    #answer = "whatever you think the answer is"
    #sent_id = "-"


    ###     End of Your Code         ###
    return sent_id, answer


def bert(question, story_id_sent, story):
    #print(question)
    #print(story_id_sent)

    #print(story)
    final_string  =''
    for dic in story:
        final_string += dic["sentence"] + ' '
    #print(final_string)
    #raise NotImplemented

    x = model_qa([final_string],[question])
    #print(x)

    answer_sent_id = '-'
    answer = '-'
    for tup in story_id_sent:
        if str(x[0][0]) in str(tup[1][0]):
            answer = tup[1][0]
            answer_sent_id = tup[0]

        
    #print(answer_sent_id)
    #print(question)
    #print(answer)
    return answer_sent_id
    #raise NotImplemented


def get_answer_parse_tree(sentenceid, sentence, question, story_const_parse, story_dep_parse, first_q_word, q_dep_parse, coref):
    answer = '-'

    #print("QUESTION", question)
    #print("SENTENCE", sentence)
    print()

    coref_dict = dict(coref)
    coref_texts = []
    for key in coref_dict:
        ls = coref_dict[key] 
        for dic in ls:
            coref_texts.append(dic["text"])
    
    question_word = get_question_word(question)
    token_pos_sent = nltk.pos_tag(nltk.word_tokenize(sentence))
    tags_no_punct = [tup for tup in token_pos_sent if re.findall(r'\w',tup[0]) != []]

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    name_entities =  [(ent.text, ent.label_) for ent in doc.ents]


    if first_q_word == 'where':
        '''print("QUESTION_DEP_PARSE")
        print(q_dep_parse)
        print()
        print("STORY_DEP_PARSE")
        print(story_dep_parse)
        print()
        print("STORY_DEP_PARSE_NODES")'''
        nodes = list(story_dep_parse.nodes.values())
        #print()

        lmtzr = WordNetLemmatizer()
        lemmatized_list = []
        for node in nodes:
            #print(node)
            tag = node["tag"]
            word = node["word"]
            if word is not None:
                if tag.startswith("V"):
                    lemmatized_list.append(lmtzr.lemmatize(word, 'v'))
                else:
                    lemmatized_list.append(lmtzr.lemmatize(word, 'n'))
        '''print(lemmatized_list)
        print()'''

        main_q_word = ''
        main_q_node = {}
        for node in q_dep_parse.nodes.values():
            if node['rel'] == 'ROOT':
                main_q_word = node["word"] 
                main_q_node = node
    
        #print("MAIN Q WORD", main_q_word)
        #print()

        lemmatized_q_word = ''
        if main_q_node["tag"].startswith("V"): 
            lemmatized_q_word = lmtzr.lemmatize(main_q_word, 'v')
        else:
            lemmatized_q_word = lmtzr.lemmatize(main_q_word, 'n')

        #print("LEMMATIZED Q WORD", lemmatized_q_word)
        #raise NotImplemented

        s_main_word = ''
        snode_dic = {}
        for node in story_dep_parse.nodes.values():
            if node["word"] is not None:
                if node["tag"].startswith("V"): 
                    lemmatized_node_word = lmtzr.lemmatize(node["word"], 'v')
                else:
                    lemmatized_node_word = lmtzr.lemmatize(node["word"], 'n')

                #print(lemmatized_node_word)
                if lemmatized_node_word is not None:
                    if lemmatized_node_word == lemmatized_q_word:
                        s_main_word = node["word"]
                        snode_dic = node 
        
        #print("STORY MAIN WORD", s_main_word)
        #print()

        if s_main_word == '':
            #print("CONST LEAVES")
            #print(question)
            question_tokens = (nltk.word_tokenize(question))
            #print(question_tokens)
            #print(story_const_parse.leaves())
            if question_tokens[len(question_tokens) - 2] in story_const_parse.leaves():
                answer = ' '.join(story_const_parse.leaves()[story_const_parse.leaves().index(question_tokens[len(question_tokens) - 2]) + 1:])
                
                if '.' in answer:
                    answer = answer.replace('.', '')
                #print(answer)
        
            #raise NotImplemented

        else:
            #print("DEPEDNENCY PARSE")
            #print(snode_dic["address"])
            #print()

            for node in story_dep_parse.nodes.values():
                if node.get('head', None) == snode_dic["address"]:
                    '''print("HEAD", node['head'])
                    print("DEPENDENTS",node["deps"])
                    print("DEPENDENCY RELATION", node['rel'])'''
                    if node['rel'] == "nmod":
                        deps = get_dependents(node, story_dep_parse)
                        deps = sorted(deps+[node], key=operator.itemgetter("address"))
                        answer = " ".join(dep["word"] for dep in deps)
    
        if answer == '-':
            #print("THIRD CASE")
            answer = process_where(tags_no_punct, name_entities, doc, question)
            if answer == None:
                mx_overlap = 0
                word_token_sent = nltk.word_tokenize(sentence)
                answer_phrase = ''
                for phrase in coref_texts:
                    if phrase in sentence:
                        word_token_phrase = nltk.word_tokenize(phrase)
                        overlap = len(set(word_token_phrase) & set(word_token_sent))
                    #print(overlap)
                        if overlap > mx_overlap:
                            mx_overlap = overlap
                            answer_phrase = phrase
                            answer = answer_phrase


    if first_q_word == 'who':
        #print("CONST_PARSE")
        #print(story_const_parse)
        #print()
        answer = process_who(sentenceid, sentence, question, story_const_parse, story_dep_parse, first_q_word)

    
    if first_q_word == 'why':
        nodes = list(story_dep_parse.nodes.values())
        #print()

        lmtzr = WordNetLemmatizer()
        lemmatized_list = []
        for node in nodes:
            #print(node)
            tag = node["tag"]
            word = node["word"]
            if word is not None:
                if tag.startswith("V"):
                    lemmatized_list.append(lmtzr.lemmatize(word, 'v'))
                else:
                    lemmatized_list.append(lmtzr.lemmatize(word, 'n'))
        '''print(lemmatized_list)
        print()'''
        question_tokenized = nltk.word_tokenize(question)
        last_question_word = question_tokenized[len(question_tokenized) - 2]

        sentence_tokens = nltk.word_tokenize(sentence)
        if last_question_word in sentence_tokens and last_question_word != sentence_tokens[len(sentence_tokens) -2]:
            #answer_phrase = " ".join
            answer_sent_1 = sentence_tokens[sentence_tokens.index(last_question_word):]
            final_answer_sent = [word for word in answer_sent_1 if re.findall(r'\w',word) != []]
            final_answer_sent.remove(final_answer_sent[0])
            answer_phrase = " ".join(final_answer_sent)
            #print("ANSWER", answer_phrase)
            return answer_phrase
        else:

            grammar3 = "INNP: {<PRP.*><VBD><PRP.*|JJ|DT|RB|MD|CD|IN|VB.*|TO>*<NN.*|RB>+<POS>?<NN.*>*}"
            cp = nltk.RegexpParser(grammar3)
            result_tree = cp.parse(tags_no_punct)
            #print(result_tree)
            result_tree_list = []
            for tup in result_tree:
                if isinstance(tup[0], tuple):
                    answer_tup = tup
                    result_tree_list.append(answer_tup)
        
            if len(result_tree_list) == 1:
                answer_tup_list = []
                #print(type(answer_tup))
                if isinstance(result_tree_list[0], nltk.tree.Tree):
                    for tup in result_tree_list[0]:
                    #print(tup)
                    #print(type(tup))
                        answer_tup_list.append(tup)

                #print(answer_tup_list)
                    if len(answer_tup_list) > 1:
                        str_tmp = ''
                        tmp_list = [str_tmp + tup[0] for tup in answer_tup_list]
                        answer_phrase = " ".join(tmp_list)
                    else:   
                        answer_phrase = answer_tup_list[0][0]
                #print(answer_phrase)
                return answer_phrase
        if answer == '-':
            main_q_word = ''
            main_q_node = {}
            for node in q_dep_parse.nodes.values():
                #print(node)
                if node['rel'] == 'ROOT':
                    main_q_word = node["word"] 
                    main_q_node = node
        
            #print("MAIN Q WORD", main_q_word)
            #print()

            lemmatized_q_word = ''
            if main_q_node["tag"].startswith("V"): 
                lemmatized_q_word = lmtzr.lemmatize(main_q_word, 'v')
            else:
                lemmatized_q_word = lmtzr.lemmatize(main_q_word, 'n')

            #print("LEMMATIZED Q WORD", lemmatized_q_word)
            #raise NotImplemented

            s_main_word = ''
            snode_dic = {}
            for node in story_dep_parse.nodes.values():
                if node["word"] is not None:
                    if node["tag"].startswith("V"): 
                        lemmatized_node_word = lmtzr.lemmatize(node["word"], 'v')
                    else:
                        lemmatized_node_word = lmtzr.lemmatize(node["word"], 'n')

                    #print(lemmatized_node_word)
                    if lemmatized_node_word is not None:
                        if lemmatized_node_word == lemmatized_q_word:
                            s_main_word = node["word"]
                            snode_dic = node 
            
            #print("STORY MAIN WORD", s_main_word)
            #print()
            
            if s_main_word == '':
                #print(story_const_parse)
                path_sim_list =[]

                for word in sentence_tokens:
                    
                    if wn.synsets(word) != [] and wn.synsets(main_q_word) !=  []:
                        path_sim = wn.synsets(word)[0].path_similarity(wn.synsets(main_q_word)[0])
                        if word not in set(nltk.corpus.stopwords.words('english')) and path_sim is not None:
                            path_sim_list.append((word, path_sim))
        
                #print(path_sim_list)
                top_words = {}
                m_word = ''
                mx = 0
                for tup in path_sim_list:
                    if tup[0] not in top_words:
                        top_words[tup[0]] = tup[1]
                        if tup[1] > mx:
                            mx = tup[1]
                            m_word = tup[0]
                
                #print(top_words)
                #print(m_word)
                answer_tmp = sentence_tokens[sentence_tokens.index(m_word) + 1:]
                answer = " ".join(answer_tmp)
                #print(answer)



            else:

                for node in story_dep_parse.nodes.values():
                    if node.get('head', None) == snode_dic["address"]:
                        '''print("HEAD", node['head'])
                        print("DEPENDENTS",node["deps"])
                        print("DEPENDENCY RELATION", node['rel'])'''
                        if node['rel'] == "nmod":
                            deps = get_dependents(node, story_dep_parse)
                            deps = sorted(deps+[node], key=operator.itemgetter("address"))
                            answer = " ".join(dep["word"] for dep in deps)
                            print(answer)

                if answer == '-':
                    answer_tmp = sentence_tokens[sentence_tokens.index(s_main_word) + 1:]
                    answer = ' '.join(answer_tmp)
                    #print(answer)
    
        #print(answer)

    
    if question_word is 'which':
        #print(story_const_parse)
        pattern = nltk.ParentedTree.fromstring("(NP)")
        subtree = pattern_matcher(pattern, story_const_parse)
        answer = " ".join(subtree.leaves())

    
    if question_word is 'did':
        #print(story_dep_parse)
        sentence_tokens = nltk.word_tokenize(sentence)
        if "not" in sentence_tokens or "no" in sentence_tokens or "never" in sentence_tokens:
            #sentence_tokens.insert(0, "No,")
            #answer_phrase = " ".join(sentence_tokens)
            answer_phrase = 'no'
            #print(answer_phrase)
            return answer_phrase

        sid = SentimentIntensityAnalyzer()
        #sid_dic = sid.polarity_scores(sentence)
        negative_score = 0
        for word in sentence_tokens:
            negative_score += sid.polarity_scores(word)['neg']
            #print(word)
            #print(sid.polarity_scores(word))
        if negative_score > 0:
            answer_phrase = 'No'
        else:
            answer_phrase = 'Yes'
        #print(answer_phrase)
        return answer_phrase
        #print()

    if question_word == 'what':
        if 'name' in question:
            answer = process_who_for_what(tags_no_punct, name_entities, doc)
            #print('poop di scooop')
            #print(answer_phrase)
        else:
            answer = process_what(tags_no_punct, name_entities, doc,question,q_dep_parse,story_dep_parse)
        
    ##Improvements:
    # How did vs How long vs How many    
    if question_word == 'how':
        answer = process_how(tags_no_punct, name_entities, doc,question,q_dep_parse,story_dep_parse, coref_texts, story_const_parse)


    if question_word == 'when':
        answer = process_when(tags_no_punct, name_entities, doc,question,story_const_parse,coref_texts)

    return answer

def process_who_for_what( tags_no_punct, name_entities, doc):
    answer_phrase = ''
    if answer_phrase == '':
            #print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
            answer_phrase = ''
            grammar = "NP: {<DT>?<JJ>*<NN>}"
            grammar2 = '''NP: {<DT|JJ|NN.*>+}
                  PP: {<IN><NP>}
                  VP: {<VB.*><NP|PP|CLAUSE>+$}
                  CLAUSE: {<NP><VP>}'''
            grammar3 = "NPVBD: {<PRP.+|DT>?<JJ>*<NN.*>+<VBD>}"

            answer_tup = ((),)
            cp = nltk.RegexpParser(grammar3)
            result_tree = cp.parse(tags_no_punct)
            #print(result_tree)
            result_tree_list = []
            for tup in result_tree:
                if isinstance(tup[0], tuple):
                    answer_tup = tup
                    #print(answer_tup)
                    break
            #print(answer_tup)
            if len(answer_tup[0]) == 0:
                for tup in name_entities:
                    if tup[1] == 'PERSON' or tup[1] == 'ORG':
                        answer_phrase = tup[0]
                        return answer_phrase
                if answer_phrase == '':
                    np_list = [chunk.text for chunk in doc.noun_chunks]
                    answer_phrase = np_list[0]
                    return answer_phrase
            
            answer_tup_list = []
            #print(type(answer_tup))
            if isinstance(answer_tup, nltk.tree.Tree):
                for tup in answer_tup:
                    answer_tup_list.append(tup)

            #print(answer_tup_list)
            answer_tup_list.remove(answer_tup_list[len(answer_tup_list) -1])

            if len(answer_tup_list) > 1:

                str_tmp = ''
                tmp_list = [str_tmp + tup[0] for tup in answer_tup_list]
                answer_phrase = " ".join(tmp_list)
            else:
                answer_phrase = answer_tup_list[0][0]

            #print(answer_phrase)
            return answer_phrase

def spacy_answer_coref(text, nlp, q_word):
    possible_when_answers = []
    for phrase in text:
        doc = nlp(phrase)
        name_entities_phrase =  [(ent.text, ent.label_) for ent in doc.ents]
        if q_word == 'when':
            for tup in name_entities_phrase:
                if tup[1] == 'CARDINAL' or tup[1] == 'DATE' or tup[1] == 'TIME':
                    #print(tup)
                    possible_when_answers.append(tup[0])
                    return possible_when_answers[0]

    return '-'


def process_what(tags_no_punct, name_entities, doc, question,q_graph,s_graph):
    #print(question)
    grammer4 =  '''NP: { <DT>?<JJ>*<CD>?<NN.*>+}'''    
    cp = nltk.RegexpParser(grammer4)
    result_tree = cp.parse(tags_no_punct)
    
    answer = doc

    tagged_q = nltk.pos_tag(nltk.word_tokenize(question))
    #print()
    #print(question)
    #print(doc)


    q_verbs = set([word if 'VB' in pos else None for word,pos in tagged_q ])
    q_nouns = set([word if 'NN' in pos else None for word,pos in tagged_q ])
    q_verbs.remove(None)
    q_nouns.remove(None)
    stopWords = nltk.corpus.stopwords.words('english')
    for x in ['did','do','does','didn\'t']:
        if x in stopWords:
            stopWords.remove(x)
    q_justwords = nltk.word_tokenize(question)
    sen_justwords = list(map(lambda x: x[0],tags_no_punct))
    ## If question is 'what did', match verb in q_verbs to verb in sentence
    ## Best verb is decided by word_vector_similarity
    ## If verbs are too similar, decide by verb position. Verbs closer to "what" word are favored
    ## Once best verb is selected in sentence, choose best noun phrase
    ## The best noun phrase follows this hierarchy
    ## 1. Rightmost noun phrase where nouns don't overlap with question nouns
    ## 2. If 1.does not exist, choose leftmost
    ## 3. If 2 does not exist, choose rightmost adjective phrase
    ## 4. If 3 does not exist, choose leftmost adjective phrase

    # changed from only applying to did to applying to all types of what questions
    # if tagged_q[1][0] == 'did':
    if True:

        def find_emergency_verb():
            if len(q_verbs) == 0:
                #print(question)
                #print("THere's no verbs here^^")
                return answer
            tagged_q.pop(1)
            verbSims = []
            for verb in q_verbs:
                bestVerb = None
                bestVerbSim = -1 * float('inf')

                vHypers = ""
                vHypos = ""
                for syn in wn.synsets(verb):
                    vHypers += " ".join([str(hy) for hy in syn.hypernyms()])
                    vHypos += " ".join([str(hy) for hy in syn.hyponyms()])
                vTotal = vHypers +' '+ vHypos
                for tup in result_tree:
                    
                    wHypers = ""
                    wHypos = ""

                    
                    

                    if not isinstance(tup[0],tuple) and 'VB' in tup[1]:
                        for syn in wn.synsets(tup[0]):
                            wHypers += " ".join([str(hy) for hy in syn.hypernyms()])
                            wHypos += " ".join([str(hy) for hy in syn.hyponyms()])
                        wTotal = wHypers + ' ' + wHypos

                        sim = cosine_similarity([sen2vec(verb)],[sen2vec(tup[0])])
                        sim += cosine_similarity([sen2vec(wTotal)],[sen2vec(vTotal)]) * .1
                        if sim > bestVerbSim:
                            bestVerb = sim
                            bestVerb = tup[0]
                    verbSims.append((bestVerbSim,verb,bestVerb))


            ## If there's no verb, something is probably wrong. 
            ## The POS tagger probably made a bad mistake
            ## In this case, just return the sentence
            

            targetVerbTup = max(verbSims, key= lambda x: x[0])
            target = targetVerbTup[2]
            verbSims.remove(targetVerbTup)


            if len(verbSims) >= 1:
                ## If the verbs are too similar
                if abs(max(verbSims,key= lambda x: x[0])[0] - targetVerbTup[0]) < .15:
                    ## finds closest verb to start of sentence (to what word)
                    clostVerb = None
                    verbIndex = float('inf')
                    verbSims.append(targetVerbTup)
                

                    for sim,q_verb,verb in verbSims:
                        i = q_justwords.index(q_verb)
                        if i < verbIndex:
                            clostVerb = verb
                            verbIndex = i

                    target = clostVerb
                
            return target
            
        q_root_node = find_main(q_graph)
        q_root = q_root_node['word']
        q_tag = q_root_node['tag']
        
        if q_root.lower() in stopWords or (not q_root_node['tag'].startswith('V') and len(q_verbs) > 0):
            q_root = find_emergency_verb()
            q_tag = 'V'

        highestSim = -1*float('inf')
        targetVerb = None
        
        if q_root == None:
            return answer
        for word,pos in tags_no_punct:
            
            sim = cosine_similarity([sen2vec(word)],[sen2vec(q_root)])
            if sim > highestSim and word not in stopWords and pos[0] == q_tag[0]:
                highestSim = sim
                targetVerb = word
            
            

        ## Now that we finally have the closest verb, we can find the closest noun phrase

        ## Rightmost noun phrase
        #print(targetVerb)
        if(targetVerb == None):
            #print("There's been a mistake. No target verb spotted. bruh bruh bruh bruh bruh")
        
            return answer


        targetNode = find_node(targetVerb,s_graph)
       # print(q_root)
        #print(targetNode['word'])
        possWords = []
        for node in get_dependents(targetNode,s_graph):
            if node['tag'].startswith('V') or node['tag'].startswith('N')  or node['tag'].startswith('J') or node['tag'].startswith('RB'):
                if (node['word'] not in q_justwords):
                    deps = get_dependents(node, s_graph)
                    deps = sorted(deps+[node], key=operator.itemgetter("address"))
                    possWords.append((node['word'],sen_justwords.index(node['word']),node['tag']," ".join(dep["word"] for dep in deps)))
       
       
       
        for node in s_graph.nodes.values():
            if  targetNode in get_dependents(node,s_graph):
                if node['tag'].startswith('V') or node['tag'].startswith('N')  or node['tag'].startswith('J') or node['tag'].startswith('RB'):
                    if (node['word'] not in q_justwords):
                        deps = get_dependents(node, s_graph)
                        deps = sorted(deps+[node], key=operator.itemgetter("address"))
                        possWords.append((node['word'],sen_justwords.index(node['word']),node['tag']," ".join(dep["word"] for dep in deps)))


        targetVerbIndex = sen_justwords.index(targetVerb)
        # possibleNounPhrases = []
        # for tup in result_tree:
        #     if isinstance(tup[0],tuple):
        #         for finnnahslidebruhfleek in tup:
        #             if 'NN' in finnnahslidebruhfleek[1]:
        #                 if not finnnahslidebruhfleek[0] in q_nouns:
        #                     possibleNounPhrases.append((tup,sen_justwords.index(finnnahslidebruhfleek[0])))
        
        sortedPhatPhrases = sorted(possWords, key=lambda ph: ph[1]-targetVerbIndex if ph[1]-targetVerbIndex > 0 else abs(ph[1]-targetVerbIndex)*100)

        ##We now prefer the noun phrases closest to 0, prefering positive values over negative values
        
        #Put adjs' at end
        for ph in sortedPhatPhrases:
            if ph[2].startswith('J'):
                sortedPhatPhrases.remove(ph)
                sortedPhatPhrases.append(ph)
        
        #Put verbs at end
        for ph in sortedPhatPhrases:
            if ph[2].startswith('V'):
                sortedPhatPhrases.remove(ph)
                sortedPhatPhrases.append(ph)
        
        #Put rest at end
        for ph in sortedPhatPhrases:
            if not ph[2].startswith('J') and not ph[2].startswith('V') and not ph[2].startswith('N'):
                sortedPhatPhrases.remove(ph)
                sortedPhatPhrases.append(ph)    
        
        
            
        # targetNounPhrase = None
        # bestPhraseScore = float('inf')
        # #print(result_tree)
        # for phrase in sortedPhatPhrases:
        #     score = phrase[1] - targetVerbIndex
        #     if abs(score) < bestPhraseScore:
        #         if score < 0:
        #             if targetNounPhrase == None:
        #                 bestPhraseScore = abs(score)
        #                 targetNounPhrase = phrase
        #         else:
        #             bestPhraseScore = abs(score)
        #             targetNounPhrase = phrase
        # if targetNounPhrase == None:
        #     #print('something not right, no best possible noun phrase')
        #     #print(result_tree)
        #     return answer
        # finally we have arrived at the answer
        # print(sortedPhatPhrases)
        if len(sortedPhatPhrases) > 0:
            targetNounPhrase = sortedPhatPhrases[0]
            
        else :
            # print('no answer')
            return answer

        answer = targetNounPhrase[3]
        if 'do' in q_justwords:
            answer = targetVerb + ' ' + answer


        #print('answer: '+answer)
        return answer
def process_how(tags_no_punct, name_entities, doc, question,q_dep_parse, s_graph,coref_texts, story_const_parse):
   
    
    answer = doc

    tagged_q = nltk.pos_tag(nltk.word_tokenize(question))


    q_verbs = set([word if 'VB' in pos else None for word,pos in tagged_q ])
    q_nouns = set([word if 'NN' in pos else None for word,pos in tagged_q ])
    q_verbs.remove(None)
    q_nouns.remove(None)

    q_justwords = nltk.word_tokenize(question)
    sen_justwords = list(map(lambda x: x[0],tags_no_punct))

    if q_justwords[1] == 'much' or q_justwords[1] == 'many':
        
       return process_when(tags_no_punct, name_entities, doc, question,story_const_parse,coref_texts)
    else: 
        return process_what(tags_no_punct, name_entities, doc, question,q_dep_parse,s_graph)


def process_when(tags_no_punct, name_entities, doc, question,s_graph,coref):
    answer_phrase = doc
    #print()
    #print(question)

    def getParent(subtree):
        targetSub = None
        lowestSubHeight = float('inf')
        for sub in s_graph.subtrees():
            # print (sub.leaves(),subtree.leaves(),sub.height(),subtree.height())
            if set(subtree.leaves()).issubset( set(sub.leaves())) and sub.height() > subtree.height():
                if sub.height() < lowestSubHeight:
                    targetSub = sub
                    lowestSubHeight = sub.height()
        return targetSub
    
    #pattern = nltk.ParentedTree.fromstring("(WHADVP)")
    pattern = nltk.ParentedTree.fromstring("(WHADVP)")
    # # Match our pattern to the tree
    subtree = pattern_matcher(pattern, s_graph)
    #subtree = s_graph.subtrees( filter=pattern)
    if subtree != None:
       # subtree = nltk.ParentedTree.convert(subtree)
        
        answer_phrase = " ".join(getParent(subtree).leaves())
       # print(answer_phrase)
        return answer_phrase
                
        
      

    for tup in name_entities:
            if tup[1] == 'CARDINAL' or tup[1] == 'DATE' or tup[1] == 'TIME':

                answer_phrase = tup[0]
                
                for subtree in s_graph.subtrees():
                    if answer_phrase in subtree.leaves() and subtree.height() == 2:
                        answer_phrase = " ".join(getParent(subtree).leaves())
                #print(answer_phrase)
                return answer_phrase
    nlp = spacy.load("en_core_web_sm")
    answer_phrase = spacy_answer_coref(coref,nlp,'when')
    for subtree in s_graph.subtrees():
                if answer_phrase in subtree.leaves() and subtree.height() == 2:
                        answer_phrase = " ".join(getParent(subtree).leaves())
                
    #print(answer_phrase)
    return answer_phrase

### DEPENDENTS
def get_dependents(node, graph):
    results = []
    for item in node["deps"]:
        address = node["deps"][item][0]
        dep = graph.nodes[address]
        results.append(dep)
        results = results + get_dependents(dep, graph)
      
    return results

def find_main(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'ROOT':
            return node
    return None
    
def find_node(word, graph):
    for node in graph.nodes.values():
        if node["word"] == word:
            return node
    return None

### CONSTITUENCY 
### took from demo stub
def pattern_matcher(pattern, tree):
    for subtree in tree.subtrees():
        node = matches(pattern, subtree)
        if node is not None:
            return node
    return None

def process_who(sentenceid, sentence, question, const_parse, dep_parse, first_q_word):
    pattern = nltk.ParentedTree.fromstring("(NP)")

    pattern2 = nltk.ParentedTree.fromstring("(VP)")

    # # Match our pattern to the tree
    subtree = pattern_matcher(pattern, const_parse)
    answer = " ".join(subtree.leaves())

    subtree2 = pattern_matcher(pattern2, const_parse)
    answer2 = " ".join(subtree2.leaves())

    '''for tree in const_parse:
        for x in tree:
            print(type(x))
            if isinstance(x, nltk.tree.Tree):
                for y in x:
                    print("Y", y)
                    print(type(y))
            print(x)
            print()'''

    #print(subtree)
    #print(subtree)
    #print(subtree2)
    if ',' in answer:
        answer_tmp = nltk.word_tokenize(answer)
        answer = ' '.join(answer_tmp[:answer_tmp.index(',')])

    #print("ANSWER NP", answer)
    #print("ANSWER VP", answer2)

    return answer


### took from demo stub
def matches(pattern, root):
    if root is None and pattern is None: 
        return root # If both nodes are null we've matched everything so far
    elif pattern is None:                
        return root # We've matched everything in the pattern we're supposed to
    elif root is None:
        return None # there's nothing to match in the tree

    # A node in a tree can either be a string (if it is a leaf) or node
    plabel = pattern if isinstance(pattern, str) else pattern.label()
    rlabel = root if isinstance(root, str) else root.label()

    # If our pattern label is the * then match no matter what
    if plabel == "*":
        return root
    elif plabel == rlabel: # Otherwise they labels need to match
        # check that all the children match.
        for pchild, rchild in zip(pattern, root):
            match = matches(pchild, rchild) 
            if match is None:
                return None 
        return root

    return None


def lemmatized_pos_selected_overlap(question, story):
    q = question["question"]
    
    punct = set(string.punctuation)

    q_sent_token = nltk.sent_tokenize(q)
    q_word_token = [nltk.word_tokenize(word) for word in q_sent_token]
    q_word_tagged = [nltk.pos_tag(word) for word in q_word_token]
    stop_words = nltk.corpus.stopwords.words("english")
    key_question_words = set([(tup[0].lower(),tup[1]) for ls in q_word_tagged for tup in ls if tup[0].lower() not in stop_words \
         and tup[0].lower() not in punct and not '\'' in tup[0] ])
    #print(key_question_words)
    #print(q_word_tagged)

    sentence_id_sent = [(dic["sentenceid"], nltk.sent_tokenize(dic["sentence"])) for dic in story ]
    #print(sentence_id_sent)
    #story_sentences = list(map(lambda x: nltk.sent_tokenize(x), [dic["sentence"] for dic in story] ))
    #print(story_sentences)
    #sentences = [nltk.word_tokenize(word) for sent in story_sentences for word in sent ]
    #sentences_tagged = [nltk.pos_tag(ls) for ls in sentences]
    sentences = [(tup[0], nltk.word_tokenize(word),tup[1])for tup in sentence_id_sent for word in tup[1]]
    sentences_tagged = [(tup[0], nltk.pos_tag(tup[1]),tup[2])for tup in sentences]
    lemmatizer = WordNetLemmatizer()

    pos_match = {"NN":'n',"JJ":'a',"VB":'v',"RB":'r'}
    pos_match.setdefault('n')
    #key_question_words = set(map(lambda w: lemmatizer.lemmatize(w[0],pos=pos_match.get(re.match('^(..?)\w*',w[1]).group(0),'n')),key_question_words))
    stemmer = SnowballStemmer('english')
    key_question_words = set(map(lambda w: stemmer.stem(w[0]),key_question_words))
    #key_question_w_posDict = {}
    #for stem,pos in key_question_words:
    #        key_question_w_posDict[stem] = pos

    #key_question_words = set(map(lambda w: w[0],key_question_words)) 

    question_word = get_question_word(q)
    q_disambiguated = disambiguate(q)

    set_q_synsets = set(map(lambda w: w[1],q_disambiguated))
    
    set_q_synsets.remove(None)

    answers = []
    for sent in sentences_tagged: 
        key_sentence_words = set([ (tup[0].lower(),tup[1]) for tup in sent[1] if tup[0].lower() not in stop_words \
              and tup[0].lower() not in punct and not '\'' in tup[0] ])
        #key_sentence_words = set(map(lambda w: lemmatizer.lemmatize(w[0],pos=pos_match.get(re.match('^(..?)\w*',w[1]).group(0),'n')),key_sentence_words))
        key_sentence_words = set(map(lambda w: (stemmer.stem(w[0]),w[1]),key_sentence_words))
        key_sentence_w_posDict = {}
        
        for stem,pos in key_sentence_words:
            key_sentence_w_posDict[stem] = pos

        sen_disambiguated = disambiguate(sent[2][0])
        set_sen_synsets = set(map(lambda w: w[1],sen_disambiguated))
        set_sen_synsets.remove(None)
        #set_sen_synsets = set(map(lambda w: re.match()))
        # print(disambiguate('Where did Andrew and his dad go'))
        # for word, syn in sen_disambiguated:
        #     if syn is not None:
        #         print(syn)
        #         print (wn.synset('circus.n.05').definition())
        #         print (wn.synset('circus.n.05')._lexname)
        #         print(syn._pos)
        #         print(syn._lemmas)
        #         synRe = re.match('(\w+)\.\(w+)\.(\w+)',syn)

        
               

        key_sentence_words = set(map(lambda w: w[0],key_sentence_words))
        

        overlap = 0
        overlapList = (key_question_words & key_sentence_words)
        for word in overlapList:
            if "nn" in key_sentence_w_posDict[word].lower():
                overlap += .5
            elif "vb" in key_sentence_w_posDict[word].lower():
                overlap += 2.5
            elif "rb" in key_sentence_w_posDict[word].lower():
                overlap += .25
            else :
                overlap += 1

        synsetOverlap = (set_q_synsets & set_sen_synsets)
        synsetOverlap = set(filter(lambda q: q is not None, synsetOverlap))
        overlap += len(synsetOverlap)
        
        # print(key_sentence_words)

        answers.append((overlap, (sent[0],key_sentence_words)))
    
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)
    best_answer = (answers[0])[1]
    #print("answer:", " ".join(tup[0] for tup in best_answer))
    #print(best_answer) 

    # if question_word == 'why':
    #     bestAnswerIndex = 0
    #     if (answers[0][0]/len(answers[0][1]) >= .9):
    #         for i in range(len(sentence_id_sent)):
    #             if sentence_id_sent[i][0] == best_answer[0]:
    #                 bestAnswerIndex = i
    #         if bestAnswerIndex < len(sentence_id_sent):
    #             # print (sentence_id_sent[i][0])
                
    #             return sentence_id_sent[i][0]
        

    answerid = best_answer[0]
    return answerid #, " ".join(tup[0] for tup in best_answer[1])

def sen2vec(sentence):
    
    model = w2vecmodel
    stop_words = nltk.corpus.stopwords.words("english")

    punct = set(string.punctuation)
    words = [word for word in nltk.word_tokenize(sentence) if word not in stop_words \
        and word.lower() not in punct and not '\'' in word ]
    res = np.zeros(model.vector_size)
    count = 0
    for word in words:
        if word in model:
            count += 1
            res += model[word]

    if count != 0:
        res /= count

    return res

def process_where(tags_no_punct, name_entities, doc, question):
    answer_phrase = ''
    grammar3 = "INNP: {<TO|IN><PRP.+|DT>?<JJ>*<NN.*>+<POS>?<NN.*>*}"

    cp = nltk.RegexpParser(grammar3)
    result_tree = cp.parse(tags_no_punct)
    #print(result_tree)
    result_tree_list = []
    for tup in result_tree:
        if isinstance(tup[0], tuple):
            answer_tup = tup
            result_tree_list.append(answer_tup)
        
    if len(result_tree_list) == 1:
        answer_tup_list = []
        #print(type(answer_tup))
        if isinstance(result_tree_list[0], nltk.tree.Tree):
            for tup in result_tree_list[0]:
                #print(tup)
                #print(type(tup))
                answer_tup_list.append(tup)
            
            if answer_tup_list[0] != 'under':
                answer_tup_list.remove(answer_tup_list[0])

            #print(answer_tup_list)
            if len(answer_tup_list) > 1:
                str_tmp = ''
                tmp_list = [str_tmp + tup[0] for tup in answer_tup_list]
                answer_phrase = " ".join(tmp_list)
            else:   
                answer_phrase = answer_tup_list[0][0]
        #print(answer_phrase)
        return answer_phrase
    elif len(result_tree_list) > 1:
        question_tokenized = nltk.word_tokenize(question)
        last_question_word = question_tokenized[len(question_tokenized) - 2]
        #print(last_question_word)
        #print(result_tree_list)
        tmp_ls = []
        for tree in result_tree_list:
            for tup in tree:
                tmp_ls.append(tup[0])
        #print(tmp_ls)
        if last_question_word not in tmp_ls:
            if tmp_ls[0] != 'at':
                tmp_ls.remove(tmp_ls[0])
            answer_phrase = " ".join(tmp_ls)
            #print(answer_phrase)
        else:
            #print(tmp_ls)
            q_word_index = tmp_ls.index(last_question_word)
            #print(q_word_index)
            answer_phrase = " ".join(tmp_ls[q_word_index+1:])
        return answer_phrase



# changed from assignment 6
def get_question_word(question):
    question = question.lower()
    question_token = nltk.word_tokenize(question)
    #print(question_token[0])
    if "why" in question_token[0]:
        return "why"
    elif "how" in question_token[0]:
        return 'how'
    elif "when" in question_token[0]:
        return "when"
    elif 'where' in question_token[0]:
        return 'where'
    elif 'what' in question_token[0]:
        return 'what'
    elif 'who' in question_token[0]:
        return 'who'
    elif 'which' in question_token[0]:
        return 'which'    
    elif 'did' or 'does' or 'do' or 'was' in question_token[0]:
        return 'did'
    else:
        return 'did'

#############################################################
###     Dont change the code in this section
#############################################################
class QAEngine(QABase):
    @staticmethod
    def answer_question(question, story):
        sent_id, answer = get_answer(question, story)
        return (sent_id, answer)


def run_qa():
    QA = QAEngine()
    QA.run()
    QA.save_answers()

#############################################################


def main():
    run_qa()

if __name__ == "__main__":
    main()

