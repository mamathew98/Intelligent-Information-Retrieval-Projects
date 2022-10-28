import networkx as nx
import matplotlib.pyplot as plt
import time


from collections import defaultdict




def main():
    lines = []
    with open('./Q2 - G2/spider10k.txt') as f:
        lines = f.readlines()

    def def_value_arr():
        return []

    def def_value_zero():
        return None

    adj_list = defaultdict(def_value_arr)
    deg_arr = defaultdict(def_value_zero)

    for line in lines:
        if line[0] == "#":
            continue
        else:
            edge = line.split()
            adj_list[int(edge[1])].append(int(edge[0]))

            if deg_arr[int(edge[0])] == None:
                deg_arr[int(edge[0])] = 1
            else:
                deg_arr[int(edge[0])] += 1
            if deg_arr[int(edge[1])] == None:
                deg_arr[int(edge[1])] = 0

    # using enumerate()
    # to find indices for 0
    ends = [n for n, d in deg_arr.items() if d == 0]

    crawl = True
    current_ends = ends
    tmp = []

    while crawl:
        # print(current_ends)
        for curr_node in current_ends:
            for nxt_node in adj_list[curr_node]:
                deg_arr[nxt_node] -= 1
                if deg_arr[nxt_node] == 0:
                    tmp.append(nxt_node)
        if len(tmp) == 0:
            crawl = False
        else:
            current_ends = tmp
            ends.extend(tmp)
            tmp.clear()


    G = nx.DiGraph()

    edges = []
    for line in lines:
        if line[0] == "#":
            continue
        else:
            edge_data = line.split()
            edges.append((edge_data[0], edge_data[1]))

    for edge in edges:
        G.add_edge(edge, 1)
    for node in ends:
        if G.has_node(node):
            G.remove_node(node)
    pr = nx.pagerank(G)
    # print(pr)
    for k, v in pr.items():
        print(k, "\t", v)


start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))
