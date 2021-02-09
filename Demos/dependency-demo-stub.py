#!/usr/bin/env python

import re, sys, nltk, operator
from nltk.stem.wordnet import WordNetLemmatizer
from qa_engine.base import QABase
    
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
    
def get_dependents(node, graph):
    results = []
    for item in node["deps"]:
        address = node["deps"][item][0]
        dep = graph.nodes[address]
        results.append(dep)
        results = results + get_dependents(dep, graph)
      
    return results


def find_answer(qgraph, sgraph):
    qmain = find_main(qgraph)
    qword = qmain["word"]
    print("qword:{}".format(qword))
    snode = find_node(qword, sgraph)
    print('snode:{}'.format(snode))

    for node in sgraph.nodes.values():
        if node.get('head', None) == snode["address"]:
            print("===node=====:{}".format(node))
            if node['rel'] == "nmod":
                deps = get_dependents(node, sgraph)
                deps = sorted(deps+[node], key=operator.itemgetter("address"))
                return " ".join(dep["word"] for dep in deps)

    #raise NotImplemented

    return None


if __name__ == '__main__':
    driver = QABase()

    # Get the first question and its story
    q = driver.get_question("fables-01_Q1")
    print("question:", q["question"])
    qgraph = q['dep_parse']

    #print(qgraph)
    #raise NotImplemented 

    story = driver.get_story(q["storyid"])
    print("sentence selected: ", story[0]['sentence'])

    sgraph = story[0]['dep_parse']
    nodes = list(sgraph.nodes.values())

    #print(nodes)
    #raise NotImplemented

    # for j, node in enumerate(nodes):
    #     print("\nNode:", j)
    #     for k,v in node.items():
    #         print("{}: {}".format(k, v))


    lmtzr = WordNetLemmatizer()
    for node in sgraph.nodes.values():
        tag = node["tag"]
        word = node["word"]
        if word is not None:
            if tag.startswith("V"):
                print(lmtzr.lemmatize(word, 'v'))
            else:
                print(lmtzr.lemmatize(word, 'n'))
    print()

    #raise NotImplemented

    answer = find_answer(qgraph, sgraph)
    print("answer:", answer)

