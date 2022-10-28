def AnswerSpan(documents):
    nz_mtx = []
    output = []
    base = documents[0].split()
    # print(base)
    for sentence in documents[1:]:
        row = sentence.split()
        both = set(base).intersection(row)
        indices_B = sorted([row.index(x) for x in both], reverse=True)
        if len(both) > 0:
            output.append(abs(indices_B[-1] - indices_B[0]))
        else:
            output.append(0)
    return output
