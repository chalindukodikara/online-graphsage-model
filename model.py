import gc
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
import tensorflow as tf
from tensorflow import keras
import random as python_random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(42)
np.random.seed(42)
python_random.seed(42)


class Model:

    def __init__(self, nodes, edges):
        self.model = None

        self.nodes = nodes
        self.edges = edges

    def initialize(self, **hyper_params):

        if (not "batch_size" in hyper_params.keys()):
            batch_size = 40
        if (not "layer_sizes" in hyper_params.keys()):
            num_samples = [20, 10]
        if (not "num_samples" in hyper_params.keys()):
            layer_sizes = [15, 15]
        if (not "bias" in hyper_params.keys()):
            bias = True
        if (not "dropout" in hyper_params.keys()):
            dropout = 0.1
        if (not "lr" in hyper_params.keys()):
            lr = 1e-2

        test_edges = self.edges.iloc[int(self.edges.shape[0] * 0.6):]
        if 'weight' in test_edges.columns:
            test_edges = test_edges.drop(['weight'], axis=1)

        graph = sg.StellarGraph(nodes=self.nodes, edges=self.edges)

        # Test split
        edge_splitter_test = EdgeSplitter(graph)
        graph_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
            p=0.3, method="global", keep_connected=False, seed=2023
        )

        indices = []
        for i in range(edge_ids_test.shape[0]):
            if not ((int(edge_ids_test[i][0]) in test_edges['source'].values) and (
                    int(edge_ids_test[i][1]) in test_edges['target'].values)):
                indices.append(i)

        edge_ids_test = np.delete(edge_ids_test, indices, axis=0)
        edge_labels_test = np.delete(edge_labels_test, indices)

        # Train split
        edge_splitter_train = EdgeSplitter(graph_test)
        graph_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
            p=0.3, method="global", keep_connected=False, seed=2023
        )

        indices = []
        for i in range(edge_ids_train.shape[0]):
            if ((int(edge_ids_train[i][0]) in test_edges['source'].values) and (
                    int(edge_ids_train[i][1]) in test_edges['target'].values)):
                indices.append(i)

        edge_ids_train = np.delete(edge_ids_train, indices, axis=0)
        edge_labels_train = np.delete(edge_labels_train, indices)

        # Train iterators
        train_gen = GraphSAGELinkGenerator(graph_train, batch_size, num_samples, weighted=False, seed=42)

        # Test iterators
        test_gen = GraphSAGELinkGenerator(graph_test, batch_size, num_samples, weighted=False, seed=42)

        # Model defining - Keras functional API + Stellargraph layers
        graphsage = GraphSAGE(
            layer_sizes=layer_sizes, generator=train_gen, bias=bias, dropout=dropout
        )

        x_inp, x_out = graphsage.in_out_tensors()

        prediction = link_classification(
            output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
        )(x_out)

        self.model = keras.Model(inputs=x_inp, outputs=prediction)

        self.model.compile(
            optimizer=keras.optimizers.Adam(lr=lr),
            loss=keras.losses.binary_crossentropy,
            metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Recall(), keras.metrics.AUC(),
                     keras.metrics.Precision()],
        )

        del test_gen
        del train_gen
        del test_edges
        del graph
        del edge_splitter_test
        del edge_labels_test
        del graph_test
        del indices
        del edge_splitter_train
        del graph_train
        del edge_labels_train
        del graphsage
        gc.collect()

        # return number of training and testing examples
        return edge_ids_train.shape[0], edge_ids_test.shape[0]

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def evaluate_one_edge_without_inputs(self):
        edges = pd.read_csv('data/china_edges.csv')
        nodes = pd.read_csv('data/china_nodes.csv', index_col=0)

        graph = sg.StellarGraph(nodes=nodes, edges=edges)
        test_gen = GraphSAGELinkGenerator(graph, 40, [20, 10], weighted=False, seed=42)
        del edges
        del nodes
        gc.collect()

        edge_ids = np.array([[2163170, 204177], [20, 78105], [2, 3]])
        edge_labels = np.array([1, 1, 0])

        x = test_gen.flow(edge_ids, edge_labels, shuffle=False)
        y = self.model.predict(x, verbose=0)
        y = y.tolist()

        return y

    # 53 columns with one node id, nodes = 54
    def evaluate_one_edge(self, nodes_list):
        edges = pd.read_csv('data/china_edges.csv')
        nodes = pd.read_csv('data/china_nodes.csv', index_col=0)

        nodes.loc[1000000000] = nodes_list[0]
        nodes.loc[1000000001] = nodes_list[1]

        # nodes = nodes.append(pd.DataFrame([x[1:]], index=[x[0]], columns=nodes.columns))

        graph = sg.StellarGraph(nodes=nodes, edges=edges)
        test_gen = GraphSAGELinkGenerator(graph, 40, [20, 10], weighted=False, seed=42)

        edge_ids = np.array([[1000000000, 1000000001]])
        edge_labels = np.array([1])

        test_set = test_gen.flow(edge_ids, edge_labels, shuffle=False)
        prediction = self.model.predict(test_set, verbose=0)
        prediction = prediction.tolist()[0][0]

        del edges
        del nodes
        del edge_ids
        del edge_labels
        del test_set
        gc.collect()

        return round(prediction, 3)

if __name__ == "__main__":

    print("")

