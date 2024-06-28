import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.utils.np_utils import to_categorical
import keras.datasets
import tensorflow as tf
from scipy import stats

"""
Постройте нейронную сеть (берем несложную полносвязную сеть, меняем число слоев, число нейронов, типы активации, 
тип оптимизатора) на датасете from sklearn.datasets import load_boston*.
"""
set_name = 'boston_house_prices.csv'
pd.set_option('display.max_columns', None)
case_dataset = pd.read_csv(set_name, header=1)
# print(case_dataset.describe())
set_col = case_dataset.columns.tolist()
y_lab = case_dataset.columns[-1]
set_col.remove(y_lab)
x_lab = set_col
y = case_dataset[y_lab]
X = case_dataset[x_lab]
z = stats.zscore(case_dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)
case_split_set = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}  # ряд значений
case_layers = [2, 3, 5]                                                    # ряд количества слоев для многослойной сети
case_neurons = [256, 4096]                                                 # ряд количества нейронов для двухмерной сети
case_models = [Sequential()]                                               # ряд типов моделей
case_activations = ['relu', 'softmax']                                     # ряд активаторов
case_optimisers = [  # keras.optimizers.adagrad_v2.Adagrad(),              # ряд оптимизаторов
                   keras.optimizers.rmsprop_v2.RMSprop(), keras.optimizers.adam_v2.Adam()]
case_scaler = [StandardScaler(), RobustScaler()]                           # ряд модулей стандартизации
case_batch_sizes = [256, 512]                                              # ряд размеров бачей
case_epoch_qt = [10, 20]                                                   # ряд количества эпох
cases_result_compare = pd.DataFrame(columns=('Layers', 'Neurons', 'Activator', 'Optimiser', 'Scaler', 'Model', 'Batch',
                                             'Epoch', 'accuracy', 'val_accuracy', 'loss', 'val_loss'))


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()


def multy_layers_model_generations(c_layer, c_neuron, c_activator, c_optimiser, c_scaler, c_batch, c_epoch,
                                   c_split_set):
    first_neuron_qt = c_neuron
    tf.random.set_seed(1)
    curr_model = Sequential()
    cur_scaler = c_scaler.fit(c_split_set['X_train'])
    x_train_s = cur_scaler.transform(c_split_set['X_train'])
    x_test_s = cur_scaler.transform(c_split_set['X_test'])
    curr_model.add(Dense(first_neuron_qt, input_shape=(len(c_split_set['X_train'].columns),), activation='relu'))
    for lay_qt in range(c_layer - 2):
        factorization_c = c_layer - lay_qt  # определяем степень для расчета количества нейронов в слое.
        neurons_qt = round(12 * (2 ** factorization_c))  # количество нейронов в зависимости от слоя
        curr_model.add(Dense(neurons_qt, input_shape=(len(c_split_set['X_train'].columns),), activation=c_activator))
    curr_model.add(Dense(1, activation=c_activator))
    curr_model.compile(loss='mean_absolute_error', optimizer=c_optimiser, metrics=['accuracy'])
    history = curr_model.fit(x_train_s, c_split_set['y_train'],
                             epochs=c_epoch,
                             batch_size=c_batch,
                             verbose=0,
                             validation_data=(x_test_s, c_split_set['y_test'])
                             )
    # print(curr_model.summary())
    # plot_loss(history)
    m_train_acc = max(history.history['accuracy'])
    m_val_acc = max(history.history['val_accuracy'])
    m_train_val = min(history.history['loss'])
    m_val_val = min(history.history['val_loss'])
    cases_result_compare.loc[len(cases_result_compare.index)] = [c_layer, c_neuron, c_activator,
                                                                 c_optimiser.__class__.__name__,
                                                                 c_scaler.__class__.__name__,
                                                                 curr_model.__class__.__name__, c_batch, c_epoch,
                                                                 m_train_acc, m_val_acc, m_train_val, m_val_val]


for case_layer in case_layers:  # Смотрим все комбинации. В дальнейшем можно использовать для сравнения любых сетей.
    for case_neuron in case_neurons:
        for case_activator in case_activations:
            for scaler in case_scaler:
                for case_optimiser in case_optimisers:
                    for case_batch_size in case_batch_sizes:
                        for case_epoch in case_epoch_qt:
                            for c_model in case_models:
                                multy_layers_model_generations(case_layer,
                                                               case_neuron,
                                                               case_activator,
                                                               case_optimiser,
                                                               scaler,
                                                               case_batch_size,
                                                               case_epoch,
                                                               case_split_set
                                                               )

print(cases_result_compare.sort_values('val_loss').head(15))
