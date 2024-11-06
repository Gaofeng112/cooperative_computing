import re
import os
def split_successors(file_path, successors):
    f=open(file_path,"r")
    lines = f.readlines()
    for line in lines:
        successor = re.split(':|;',line)
        del successor[0]
        del successor[-1]
        successors.append(successor)
    for i in range(len(successors)):
        for successor in successors[i]:
            if str(i) in successors[int(successor)]:
                text = "error:"+str(i)+','+successor
                print(text)
successors = []
split_successors('successor_final_3.txt',successors)
print('end')    

import networkx as nx
import matplotlib.pyplot as plt 
G = nx.DiGraph() 
G.add_nodes_from(range(len(successors))) 
for i in range(len(successors)): 
    for successor in successors[i]: 
        G.add_edge(i,int(successor))
pos = nx.spring_layout(G)
con = nx.strongly_connected_components(G)
#print(con,type(con),list(con))
lst = list(con)
for elem in lst:
    if(len(elem) >1):
        print(elem)
order = list(nx.topological_sort(G))
print(order)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=50, font_size= 6)
plt.show()    