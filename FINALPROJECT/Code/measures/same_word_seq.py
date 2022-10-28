def lcs_iterative_space_optim(A, B, i=0, j=0):
    X = [0] * (len(A) + 1)
    Y = [0] * (len(B) + 1)

    if len(X) >= len(Y):
        m = min(len(Y), len(X))
        for i in range(len(A) - 1, -1, -1):
            for j in range(len(B) - 1, -1, -1):
                if A[i] == B[j]:
                    X[j] = 1 + Y[j + 1]

                else:
                    X[j] = max(Y[j], X[j + 1])

            for n in range(m):
                Y[n] = X[n]

        return X[0]
    else:
        return lcs_iterative_space_optim(B, A)


def SameWordSeq(documents):
    nz_mtx = []
    output = []
    base = documents[0].split()
    # print(base)
    for sentence in documents[1:]:
        row = sentence.split()
        # print(row)
        output.append(lcs_iterative_space_optim(base, row))
    return output
