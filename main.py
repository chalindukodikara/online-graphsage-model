import gc
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def test():
    edges = pd.read_csv('data/china_edges.csv')
    nodes = pd.read_csv('data/china_nodes.csv', index_col=0)

    from model import Model

    MODEL_TEST = Model(nodes, edges)
    MODEL_TEST.initialize()

    del edges
    del nodes
    gc.collect()

    # we should be able to give node and its attributes from outside
    print("############################## TEST STARTED ##############################")

    for i in range(len(WEIGHTS)):
        final_model_weights = np.load(WEIGHTS[i], allow_pickle=True)
        MODEL_TEST.set_weights(final_model_weights)
        print(WEIGHTS[i].split('/')[1].split('.')[0], "=", MODEL_TEST.evaluate_one_edge_without_inputs())

    print("############################## TEST FINISHED ##############################")

def main():
    edges = pd.read_csv('data/china_edges.csv')
    nodes = pd.read_csv('data/china_nodes.csv', index_col=0)

    from model import Model

    MODEL = Model(nodes, edges)
    MODEL.initialize()

    del edges
    del nodes
    gc.collect()

    # node list that we have to feed into the function
    nodes_list = [[0.1 for i in range(53)], [0.15 for i in range(53)]]

    # select the model you want to train
    final_model_weights = np.load(WEIGHTS[1], allow_pickle=True)
    MODEL.set_weights(final_model_weights)

    print("############################## TEST FOR ONE EDGE ##############################")
    print("LINK PROBABILITY: ", round(float(MODEL.evaluate_one_edge(nodes_list=nodes_list))*100, 1), "%")


if __name__ == "__main__":

    WEIGHTS = ["weights/dblp_china_pc_1.npy", "weights/dblp_usa_pc_1.npy", "weights/dblp_germany_pc_1.npy"
        , "weights/dblp_china_pc_2.npy", "weights/dblp_usa_pc_2.npy", "weights/dblp_germany_pc_2.npy",
               "weights/federated_pc_2.npy"]

    test()

    main()