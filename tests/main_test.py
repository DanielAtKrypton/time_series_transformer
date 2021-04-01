"""
main test script
To run issue the command pytest at the root folder of the project.
"""
import time
from datetime import timedelta

import pytest
from flights_time_series_dataset import FlightsDataset, FlightSeriesDataset
from time_series_predictor import TimeSeriesPredictor
from time_series_transformer import Transformer
from skorch.dataset import CVSplit
from skorch.callbacks import EarlyStopping
from torch.optim import Adam
import numpy as np
from sklearn.metrics import r2_score
from .helpers import cuda_check
from .config import devices

@pytest.mark.parametrize('device', devices)
def test_transformer_tsp(device):
    cuda_check(device)

    start = time.time()
    tsp = TimeSeriesPredictor(
        Transformer(),
        max_epochs=50,
        train_split=None,
        device=device,
    )

    tsp.fit(FlightsDataset())
    score = tsp.score(tsp.dataset)
    assert score > -1
    end = time.time()
    elapsed = timedelta(seconds = end - start)
    print("Fitting in {} time delta: {}".format(device, elapsed))

@pytest.mark.parametrize('device', devices)
def test_transformer_tsp_multisamples(device):
    cuda_check(device)

    start = time.time()
    tsp = TimeSeriesPredictor(
        Transformer(d_model=12),
        lr = 1e-5,
        lambda1=1e-8,
        optimizer__weight_decay=1e-8,
        iterator_train__shuffle=True,
        early_stopping=EarlyStopping(patience=100),
        max_epochs=500,
        train_split=CVSplit(10),
        optimizer=Adam,
        device=device,
    )

    past_pattern_length = 24
    future_pattern_length = 12
    pattern_length = past_pattern_length + future_pattern_length
    fsd = FlightSeriesDataset(pattern_length, future_pattern_length, pattern_length, stride = 1, generate_test_dataset=True)
    tsp.fit(fsd)
    end = time.time()
    elapsed = timedelta(seconds = end - start)
    print("Fitting in {} time delta: {}".format(device, elapsed))

    mean_r2_score = tsp.score(tsp.dataset)
    assert mean_r2_score > 0

    netout = tsp.predict(fsd.test.x)

    idx = np.random.randint(0, len(fsd.test.x))

    y_true = fsd.test.y[idx, :, :]
    y_hat = netout[idx, :, :]
    r2s = r2_score(y_true, y_hat)
    assert r2s > -1
    print("Final R2 score: {}".format(r2s))