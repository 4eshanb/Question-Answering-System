#!/usr/bin/env python


import sys, nltk
from nltk.tree import Tree

from qa_engine.base import QABase


# See if our pattern matches the current root of the tree
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
    
def pattern_matcher(pattern, tree):
    for subtree in tree.subtrees():
        node = matches(pattern, subtree)
        if node is not None:
            return node
    return None


def main():
    driver = QABase()
    q = driver.get_question("fables-01_Q1")
    story = driver.get_story(q["storyid"])
    print("sentence selected:{}".format(story))

    tree = story[0]['const_parse']
    print("const tree:{}".format(tree))

    # Create our pattern
    pattern = nltk.ParentedTree.fromstring("(VP (*) (PP))")

    # # Match our pattern to the tree
    subtree = pattern_matcher(pattern, tree)
    print(" ".join(subtree.leaves()))


    # create a new pattern to match a smaller subset of subtree
    pattern2 = nltk.ParentedTree.fromstring("(PP)")

    # Find and print the answer
    subtree2 = pattern_matcher(pattern2, subtree)
    print(" ".join(subtree2.leaves()))

if __name__ == '__main__':
    main()

