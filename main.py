from graph import build_hetero_graph

g = build_hetero_graph('test.pcap')

# StellarGraph still does not support Edge Graphs
# i remember (one) of my promblems now
# Ill just use something else instead
# Wanna try something new, (and mainstream)
from stellargraph import StellarGraph
graphs = []
graphs.append(StellarGraph.from_networkx(g, edge_type_attr="feature"))

from stellargraph.mapper import PaddedGraphGenerator
generator = PaddedGraphGenerator(graphs=graphs)

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarGraph

from stellargraph import datasets

from sklearn import model_selection
from IPython.display import display, HTML

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf

# ////////////// KERAS MODEL ///////////////////
print('Creating keras model')
k = 35  # the number of rows for the output tensor
layer_sizes = [32, 32, 32, 1]

dgcnn_model = DeepGraphCNN(
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "tanh"],
    k=k,
    bias=False,
    generator=generator,
)
x_inp, x_out = dgcnn_model.in_out_tensors()

x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
x_out = MaxPool1D(pool_size=2)(x_out)

x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

x_out = Flatten()(x_out)

x_out = Dense(units=128, activation="relu")(x_out)
x_out = Dropout(rate=0.5)(x_out)

predictions = Dense(units=1, activation="sigmoid")(x_out)

model = Model(inputs=x_inp, outputs=predictions)

model.compile(
    optimizer=Adam(lr=0.0001), loss=binary_crossentropy, metrics=["acc"], # type: ignore
)

# ////////////// TRAIN //////////////////
graph_labels = [0]
print('training')
train_graphs, test_graphs = model_selection.train_test_split(
    graph_labels, train_size=0.5, test_size=2, stratify=graph_labels,
)

gen = PaddedGraphGenerator(graphs=graphs)

train_gen = gen.flow( # should this be gen or generator
    list(train_graphs.index - 1),
    targets=train_graphs.values,
    batch_size=50,
    symmetric_normalization=False,
)

test_gen = gen.flow(
    list(test_graphs.index - 1),
    targets=test_graphs.values,
    batch_size=1,
    symmetric_normalization=False,
)

epochs = 100

history = model.fit( # changing verbose over 2 removes cool ui
    train_gen, epochs=epochs, verbose=2, validation_data=test_gen, shuffle=False,
)
