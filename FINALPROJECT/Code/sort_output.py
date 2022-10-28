import pandas as pd
import scipy.stats as ss


def sort_output(f_type, file="w2v_out.csv"):
    df = pd.read_csv(file)
    questions = df["0"]
    answers = df["1"]
    labels = df["2"]
    points = df["3"]
    ranks = []
    for i in range(int((len(points) / 10) + 1)):
        sel = points[i*10: (i+1)*10]
        # rankdata = ss.rankdata(sel)
        rank = len(sel) - ss.rankdata(sel).astype(int) + 1
        # rank = [int(x) for x in rankdata]
        ranks.extend(rank)
    # print(ranks)
    out = pd.DataFrame(zip(questions, answers, ranks, points, labels), columns=[
        "Question", "Answer", "Rank", "Point", "Judge"
    ])
    out.to_csv('OUTPUT_{}.tsv'.format(f_type), sep="\t", index=False, header=[
        "Question", "Answer", "Rank", "Point", "Judge"
    ])
    # print(out)


if __name__ == '__main__':
    sort_output(f_type="WORD2VEC")
    sort_output("MLP", 'mlp_test.csv')
