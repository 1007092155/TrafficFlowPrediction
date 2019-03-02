"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.dataNew import process_data
from model import modelNew
from keras.models import Model
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")


def train_model(model, x_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        x_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    model.compile(loss="mean_squared_error", optimizer="Nadam", metrics=['mae'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        x_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.02)

    model.save('model/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + name + ' loss.csv', encoding='utf-8', index=False)


def train_seas(models, x_train, y_train, name, config):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        x_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = x_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(input=p.input,
                                       output=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, x_train, y_train, name, config)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstmNew",
        help="Model to train.")
    args = parser.parse_args()

    lag = 6
    config = {"batch": 256, "epochs": 400}
    file1 = 'data/train_data_cluster0(after6am).csv'
    file2 = 'data/test_data_cluster0(after6am).csv'
    x_train, y_train, _, _, _ = process_data(file1, file2, lag)

    if args.model == 'lstmNew':
        x_train = np.reshape(x_train, (x_train.shape[0], lag, 6))
        m = modelNew.get_lstm([6, 6, 20, 20, 1])
        train_model(m, x_train, y_train, args.model, config)
    if args.model == 'gru':
        x_train = np.reshape(x_train, (x_train.shape[0], lag, 12))
        m = modelNew.get_gru([7, 12, 60, 75, 1])
        train_model(m, x_train, y_train, args.model, config)
    if args.model == 'saes':
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
        m = modelNew.get_saes([12, 400, 400, 400, 1])
        train_seas(m, x_train, y_train, args.model, config)


if __name__ == '__main__':
    main(sys.argv)
