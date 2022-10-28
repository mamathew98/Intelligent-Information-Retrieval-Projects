import networkx as nx
import matplotlib.pyplot as plt
import time


def main():
    G = nx.DiGraph()

    lines = []
    with open('./Q2 - G2/spider800k.txt') as f:
        lines = f.readlines()

    edges = []
    for line in lines:
        if line[0] == "#":
            continue
        else:
            edge_data = line.split()
            edges.append((edge_data[0], edge_data[1]))

    G.add_edges_from(edges)
    pr = nx.pagerank(G, max_iter=10)
    # print(pr)
    for k, v in pr.items():
        print(k, "\t", v)


start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))