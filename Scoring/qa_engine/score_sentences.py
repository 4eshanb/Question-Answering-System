import pandas as pd
import argparse

def score_all_answers(gold, pred, incorrect_only=True):
    all_scores = []
    if len(gold) != len(pred):
        print("WARNING: you are missing questions in your response file.")
        exit(1)
    for row in gold.itertuples():
        print("-"*40)

        gold_answer = str(row.answer_sentenceid.strip())
        pred_answer = str(pred.loc[row.Index].answer_sentenceid).strip()
        correct = gold_answer == pred_answer

        if incorrect_only and not correct:
            print('Your answer is INCORRECT')
            print('Comparing questionid: {}. Gold answer: {}. Your answer: {}'.format(row.Index, gold_answer, pred_answer))

        all_scores.append(int(correct))

    print("-" * 40)
    return sum(all_scores)/len(gold)


def run_scoring(gold, pred):
    accuracy = score_all_answers(gold, pred)

    print("FINAL RESULTS\n\n")

    print("AVERAGE ACCURACY =     {:.4f}".format(accuracy))
    print("\n*************************************************************************\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assignment 6')
    parser.add_argument('-answer', dest='answer_fname', help='The path to answer key file')
    parser.add_argument('-response', dest='response_fname', help='The path to your response file')
    args = parser.parse_args()

    gold = pd.read_csv(args.answer_fname, index_col="questionid", sep="\t")
    pred = pd.read_csv(args.response_fname, index_col="questionid", sep="\t")
    run_scoring(gold, pred)


