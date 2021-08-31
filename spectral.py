import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
import scipy


class Point():
    def __init__(self, x, y, color="blue"):
        self.x = x
        self.y = y
        self.color = color

def plot(points):
    for point in points:
        plt.plot(point.x, point.y, 'o', color=point.color)
    plt.show()

def graph_plot(graph):
    G = nx.from_numpy_matrix(np.matrix(graph.round(3)), create_using=nx.Graph)
    layout = nx.spring_layout(G)
    nx.draw(G, layout, node_color="blue")
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=labels)
    plt.show()

def plot_eigs(eigs):
    for i, eig in enumerate(eigs):
        plt.plot(eig[0], eig[1], 'o', color="blue")
    plt.show()

# Not used in example, but verbose way of defining similarity
def similarity(a, b):
    vec_sub = np.array([a.x, a.y]) - np.array([b.x, b.y])
    sim = np.exp(-1 * np.square(np.linalg.norm(vec_sub)))
    return round(sim, 4)

# Not used in example, but verbose way of generating graph
def create_graph(points):
    graph = np.zeros((len(points), len(points)))
    for row in range(len(graph)):
        for col in range(len(graph)):
            graph[row][col] = similarity(points[row], points[col])
    return graph
    
def degree_matrix(graph):
    return np.diag(np.sum(graph, axis=1))

def laplacian_eigenvectors(graph):
    degree = degree_matrix(graph)
    laplacian = degree - graph
    _, eigenvectors = scipy.linalg.eigh(laplacian, subset_by_index=[0, 1])
    return eigenvectors

def find_clusters(graph, points):
    eigs = laplacian_eigenvectors(graph)
    plot_eigs(eigs)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(eigs)
    for i in range(len(kmeans.labels_)):
        if kmeans.labels_[i] == 0:
            points[i].color = "green"
        else:
            points[i].color = "red"
    plot(points)


def main():
    dataset = [Point(2, 0), Point(4,  0), Point(6, 0), Point(0, 2), 
               Point(0, 4), Point(0, 6), Point(0, 8), Point(2, 10), Point(4, 10),
               Point(6, 10), Point(4, 5), Point(6, 5), Point(8, 5), Point(10, 5)]

    plot(dataset)

    points = [[point.x, point.y] for point in dataset]
    graph = kneighbors_graph(points, n_neighbors=3, mode="distance").toarray()

    graph_plot(graph)

    find_clusters(graph, dataset)

    


if __name__ == "__main__":
    main()