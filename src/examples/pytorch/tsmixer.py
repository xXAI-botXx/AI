"""
This file contains a implementation of the TSMixer model in PyTorch
"""

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from datetime import datetime as dt
import pickle


#self.activation = getattr(nn, activation)()

activation_functions = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "laekyrelu": nn.LeakyReLU
}

class ResBlock(nn.Module):
    """
    Residual block for TSMixer.

    Parameters:
    - norm_type (str): Type of normalization, either 'L' for LayerNorm or 'B' for BatchNorm.
    - activation (str): Activation function to be used.
    - dropout (float): Dropout rate for regularization.
    - ff_dim (int): Dimension of the feature linear layer.

    Attributes:
    - norm: Normalization layer.
    - temporal_linear: Temporal linear layer.
    - feature_linear: Feature linear layer.
    """

    def __init__(self, input_shape, norm_type, activation, dropout, ff_dim):
        super(ResBlock, self).__init__()

        # define normalization
        self.norm = (
            nn.LayerNorm
            if norm_type == 'L'
            else nn.BatchNorm1d
        )

        # Temporal Linear
        self.temporal_linear = nn.Sequential(
            self.norm(input_shape[-2]),  
            nn.Linear(input_shape[-2], input_shape[-2]), 
            activation_functions[activation](),
            nn.Dropout(dropout)
        )
        
        # Feature Linear
        self.feature_linear = nn.Sequential(
            self.norm(input_shape[-1]),   
            nn.Linear(input_shape[-1], ff_dim),    
            activation_functions[activation](),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_shape[-1]),   
            nn.Dropout(dropout)
        )

    def forward(self, inputs):
        """
        Forward pass through the residual block.

        Parameters:
        - inputs (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        x = inputs
        # Temporal Linear
        x = x.transpose(1, 0)  # switch to -> [Channel, Input Length]
        x = self.temporal_linear(x)
        x = x.transpose(1, 0)  # switch to -> [Input Length, Channel]
        res = x + inputs

        # Feature Linear
        #x = res.transpose(1, 2)  # [Input Length, Channel]
        x = self.feature_linear(x)
        #x = x.transpose(1, 2)  # [Channel, Input Length]

        return x + res

class TSMixerModel(nn.Module):
    """
    TSMixer model.

    Parameters:
    - input_shape (int): Input shape.
    - pred_len (int): Length of the prediction.
    - norm_type (str): Type of normalization, either 'L' for LayerNorm or 'B' for BatchNorm.
    - activation (str): Activation function to be used.
    - n_block (int): Number of residual blocks in the model.
    - dropout (float): Dropout rate for regularization.
    - ff_dim (int): Dimension of the feature linear layer.
    - target_slice (int or None): Index of the target slice or None.

    Attributes:
    - blocks: List of residual blocks.
    - target_slice: Index of the target slice.
    - output_layer: Output linear layer.
    """

    def __init__(self, input_shape, pred_len, norm_type, activation, n_block, dropout, ff_dim, target_slice):
        super(TSMixerModel, self).__init__()

        self.blocks = nn.ModuleList([
            ResBlock(input_shape, norm_type, activation, dropout, ff_dim) for _ in range(n_block)
        ])

        self.target_slice = target_slice

        self.output_layer = nn.Sequential(
            nn.Linear(input_shape[-2], pred_len)
        )

        self.output_feature_reduction_layer = nn.Sequential(
            nn.Linear(input_shape[-1], 1)
        )

    def forward(self, x):
        """
        Forward pass through the TSMixer model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        for block in self.blocks:
            x = block(x)

        if self.target_slice:
            x = x[:, self.target_slice]

        x = x.transpose(1, 0)  # [Channel, Input Length]
        x = self.output_layer(x)  # [Channel, Output Length]
        x = x.transpose(1, 0)  # [Output Length, Channel]
        x = self.output_feature_reduction_layer(x)

        return x

    def predict(self, dataloader, batch_size):
        """
        Predicts using the trained model.

        Args:
        - dataloader (DataLoader): DataLoader containing the input data.
        - batch_size (int): Batch size for processing the data.

        Returns:
        - numpy.ndarray: Predictions.
        """
        self.eval()
        prediction = None

        with torch.no_grad():
            for batch_X in dataloader:
                if type(batch_X) == list:
                    batch_X = batch_X[0] 

                batch_X, X_padding = self.correct_shape(batch_X=batch_X, batch_y=None, BATCH_SIZE=batch_size)
                
                outputs = self.forward(batch_X).squeeze().numpy()
                if X_padding > 0:
                    outputs = outputs[:-X_padding]

                outputs = np.reshape(outputs, (-1, 1))
                
                if type(prediction) == type(None):
                    prediction = outputs
                else:
                    prediction = np.vstack((prediction, outputs))

                #all_predictions += [outputs.squeeze().numpy()]

        prediction = np.reshape(prediction, (-1, ))

        return prediction

    def correct_shape(self, batch_X, batch_y, BATCH_SIZE):
        """
        Corrects the shape of input batches to ensure consistency during processing.

        Args:
        - batch_X (torch.Tensor): Input batch tensor.
        - batch_y (torch.Tensor): Target batch tensor.
        - BATCH_SIZE (int): Batch size for padding.

        Returns:
        - list: List containing corrected input and padding information.
        """
        res = []

        if type(batch_X) != type(None): 
            X_padding = max(0, BATCH_SIZE-batch_X.shape[0])
            if X_padding > 0:
                batch_X = torch.nn.functional.pad(batch_X, (0, 0, 0, X_padding), value=-99999)
            res += [batch_X, X_padding]

        if type(batch_y) != type(None):
            y_padding = max(0, BATCH_SIZE-batch_y.shape[0])
            if y_padding > 0:
                batch_y = torch.nn.functional.pad(batch_y, (0, y_padding), value=-99999)
            res += [batch_y, y_padding]

        return res



def get_short_duration_representation(start, end, should_print=False):
    """
    Returns a string representation of the duration between two datetime objects.

    Parameters:
    - start (datetime): The start datetime object.
    - end (datetime): The end datetime object.
    - should_print (bool, optional): Whether to print the duration string. Default is False.

    Returns:
    - str: A string representation of the duration in the format "xD xH xM xS".
    """
    duration = abs((start-end).total_seconds())
    minutes, seconds = divmod(duration, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    res = f"{int(days)}D {int(hours)}H {int(minutes)}M {int(seconds)}S"
    if should_print:
        print(res)
    return res



def train(train_loader, epochs=20, learning_rate=1e-4, n_block=3, ff_dim=64, 
            dropout=0.1, activation='laekyrelu', optimizer_name=optim.Adam, 
            criterion_name=nn.MSELoss, save_path="./results/tsmixer_experiment",
            experiment_name="TSMixer", should_save=False, show_res=False,
            batch_size=1440, n_features=1, common_batch_shape=[1440, 1],
            test_loader=None):
    """
    Trains the TSMixer model.

    Parameters:
    - train_loader (DataLoader): Training data loader.
    - epochs (int): Number of epochs for training. Default is 20.
    - learning_rate (float): Learning rate for optimization. Default is 1e-4.
    - n_block (int): Number of residual blocks in the model. Default is 3.
    - ff_dim (int): Dimension of the feature linear layer. Default is 64.
    - dropout (float): Dropout rate for regularization. Default is 0.1.
    - activation (str): Activation function to be used. Default is 'laekyrelu'.
    - optimizer_name (torch.optim): Optimizer class for optimization. Default is optim.Adam.
    - criterion_name (torch.nn): Criterion class for calculating loss. Default is nn.MSELoss.
    - save_path (str): Path to save experiment results. Default is "./results/tsmixer_experiment".
    - experiment_name (str): Name of the experiment. Default is "TSMixer".
    - should_save (bool): Whether to save experiment results. Default is False.
    - show_res (bool): Whether to display training results. Default is False.
    - batch_size (int): Batch size for training. Default is 1440.
    - n_features (int): Number of features. Default is 1.
    - common_batch_shape (list): Common batch shape. Default is [1440, 1].
    - test_loader (DataLoader): Test data loader. Default is None.

    Returns:
    - tuple: Tuple containing the trained model, log, total loss, and average loss.
    """
    # create TSMixer-Modell
    input_shape = [batch_size, n_features]  
    pred_len = batch_size #helper.PREDICTION_PERIOD  
    norm_type = 'L'
    target_slice = None    # not in use
    criterion = criterion_name()
    log = ""

    model = TSMixerModel(input_shape, pred_len, norm_type, activation, n_block, dropout, ff_dim, target_slice)

    optimizer = optimizer_name(model.parameters(), lr=learning_rate)

    start = dt.now()

    start_str = f'\n\n{"#"*16}\nExperiment:\n    - name = {experiment_name}\
    \n    - start = {start.strftime("%Y-%m-%d %H:%M OClock")}\n    - learn-rate = {learning_rate}\
    \n    - epochs = {epochs}\n    - n_blocks = {n_block}\n    - ff_dim = {ff_dim}\
    \n    - dropout = {dropout}\n    - activation = {activation}\
    \n    - criterion = {criterion_name.__name__}\n    - optimizer = {optimizer_name.__name__}'
    log += start_str
    if show_res:
        print(start_str)

    # training
    loss_hist = []
    steps = 0
    log = ""

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, X_padding, batch_y, y_padding = model.correct_shape(batch_X=batch_X, batch_y=batch_y, BATCH_SIZE=batch_size)

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)  
            #outputs = outputs.numpy()
            if X_padding > 0:
                outputs = outputs[:-X_padding]
            loss_hist += [loss.item()]
            loss.backward()
            optimizer.step()
            steps += 1

        epoch_str = f'-> Epoch {epoch+1}/{epochs}, Training Loss: {loss.item():.4f}, Steps: {steps}'
        log += f"\n{epoch_str}"
        if show_res:
            print(epoch_str)
    last_loss = loss.item()

    end_str = f"Training Finish, duration = {get_short_duration_representation(start=start, end=dt.now())}"
    log += f"\n\n{end_str}"
    if show_res:
        print(end_str)

    log += f"\n\nLast loss: {last_loss}"

    # draw it
    if show_res == False:
        plt.ioff()

    loss_series = pd.Series(loss_hist)
    window_size = 100 
    loss_rolling = loss_series.rolling(window=window_size).mean()

    if show_res or should_save:
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        #ax.plot(np.arange(len(loss_hist)), loss_hist, label='Original Loss')
        ax.plot(np.arange(len(loss_series.index)), loss_rolling, label='Smoothed Loss', linewidth=2)
        ax.set_xlabel('Learning progress')
        ax.set_ylabel('Loss (normalized mean absolute error)')
        ax.set_title('Loss over time')
        ax.legend()
        ax.grid()
        if show_res:
            plt.show()

        if should_save:
            plt.savefig(f"{save_path}/{experiment_name}_loss.png")

    # testing on dev-data
    if type(test_loader) != type(None):
        average_loss, total_loss = test_data(model, dataloader=test_loader, save_path=save_path, 
                                                    experiment_name=experiment_name, should_save=should_save, 
                                                    show_res=show_res, criterion_name=criterion_name,
                                                    batch_size=batch_size)
    else:
        total_loss = -999
        average_loss = -999

    log += f"\n\nAVG Dev loss: {average_loss}"

    # save experiment results
    if should_save:
        with open(f"{save_path}/{experiment_name}_log.txt", "w") as f:
            f.write(log)

    return model, log, total_loss, average_loss



def test_data(model, dataloader, save_path="./results/tsmixer_experiment",
                experiment_name="TSMixer", should_save=False, show_res=True,
                show_residuals=False, criterion_name=nn.MSELoss,
                window_size=25, batch_size=1440, return_prediction=False):
    """
    Tests the TSMixer model on given data.

    Parameters:
    - model (TSMixerModel): Trained TSMixer model.
    - dataloader (DataLoader): Data loader for testing.
    - save_path (str): Path to save experiment results. Default is "./results/tsmixer_experiment".
    - experiment_name (str): Name of the experiment. Default is "TSMixer".
    - should_save (bool): Whether to save experiment results. Default is False.
    - show_res (bool): Whether to display test results. Default is True.
    - show_residuals (bool): Whether to display residuals. Default is False.
    - criterion_name (torch.nn): Criterion class for calculating loss. Default is nn.MSELoss.
    - window_size (int): Window size for smoothing. Default is 25.
    - batch_size (int): Batch size for testing. Default is 1440.
    - return_prediction (bool): Whether to return predictions. Default is False.

    Returns:
    - tuple: Tuple containing average loss and total loss if return_prediction is False.
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    got_example_predicton = False
    example_y_pred = None
    example_y = None
    timesteps = None
    loss_hist = []
    criterion = criterion_name()
    prediction = None

    with torch.no_grad():
        amount = 0
        for batch_X, batch_y in dataloader:
            batch_X, X_padding, batch_y, y_padding = model.correct_shape(batch_X=batch_X, batch_y=batch_y, BATCH_SIZE=batch_size)

            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            outputs = outputs.numpy()
            if X_padding > 0:
                outputs = outputs[:-X_padding]
            total_loss += loss.item()
            loss_hist += [loss.item()]
                
            if type(prediction) == type(None):
                prediction = np.reshape(outputs, (-1, 1))
            else:
                prediction = np.vstack((prediction, np.reshape(outputs, (-1, 1))))

            all_predictions += [outputs]
            amount += 1

            if got_example_predicton == False:
                if np.random.rand() > 0.1:
                    example_y_pred = outputs
                    example_y = batch_y
                    time_steps = np.arange(len(batch_y))
                    got_example_predicton = True

    if got_example_predicton == False:
        example_y_pred = outputs
        example_y = batch_y
        time_steps = np.arange(len(batch_y))
        got_example_predicton = True

    prediction = np.reshape(prediction, (-1, ))


    average_loss = total_loss / amount
    if show_res:
        print(f'Average Loss: {average_loss:.4f}')
    else:
        plt.ioff()

    if show_res or should_save:
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        # plot the timeseries
        example_y = pd.Series(example_y)
        example_y_pred = pd.Series(example_y_pred)
        example_y = example_y.rolling(window=window_size).mean()
        example_y_pred = example_y_pred.rolling(window=window_size).mean()

        ax.plot(np.arange(len(example_y.index)), example_y, label='Ground Truth', marker='o')
        ax.plot(np.arange(len(example_y_pred.index)), example_y_pred, label='Predictions', marker='x', alpha=0.6)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Values')
        ax.set_title(f'Time Series Predictions on data (AVG={average_loss:.4f})')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    if show_res:
        plt.show()

    if should_save:
        plt.savefig(f"{save_path}/{experiment_name}_example_test_prediction.png")


    # residuals
    if show_residuals:
        if show_res or should_save:
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            ax.hist(loss_hist, bins=15, color='green', edgecolor='black')
            ax.set_xlabel(f'Loss ({criterion})')
            ax.set_ylabel('Amount of values with this loss')
            ax.set_title('Losses')
            ax.grid()

            if show_res:
                plt.show()

            if should_save:
                plt.savefig(f"{save_path}/{experiment_name}_residuals.png")

    if return_prediction:
        return prediction
    else:
        return average_loss, total_loss



def create_dataloader(X, y, X_test=None, batch_size=1440, shuffle=False, should_print=False):
    X_tensor = torch.tensor(X.values.astype("int"), dtype=torch.float32).float()
    y_tensor = torch.tensor(y.values.astype("int"), dtype=torch.float32).float()
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    """
    Creates a DataLoader for training and testing.

    Parameters:
    - X (pd.DataFrame): Features dataframe.
    - y (pd.Series): Target series.
    - X_test (pd.DataFrame): Test features dataframe. Default is None.
    - batch_size (int): Batch size for DataLoader. Default is 1440.
    - shuffle (bool): Whether to shuffle data. Default is False.
    - should_print (bool): Whether to print batch shapes. Default is False.

    Returns:
    - DataLoader: Training DataLoader.
    - DataLoader: Testing DataLoader if X_test is provided, otherwise None.
    - list: Common batch shape.
    - int: Number of features.
    - int: Number of batches.
    """
    if type(X_test) != type(None):
        X_test_tensor = torch.tensor(X_test.values.astype("int"), dtype=torch.float32).float()
        test_dataset = TensorDataset(X_test_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    count = 0
    batch_x_shapes = []
    for batch_X, batch_y in dataloader:
        if count == 3:
            if batch_x_shapes[0] == batch_x_shapes[1] == batch_x_shapes[2]:
                COMMON_BATCH_SHAPE = batch_x_shapes[0]
                N_FEATURES = batch_x_shapes[0][-1]
            else:
                raise ValueError("The batches have another shape! Try to run the code again.")
        if count < 3:
            if should_print:
                print("X:", batch_X.shape)
            batch_x_shapes += [batch_X.shape]
            if should_print:
                print("y:", batch_y.shape, "\n")
        else:
            if batch_X.shape != COMMON_BATCH_SHAPE and should_print:
                print("Warning:", batch_X.shape, "On batch count:", count)
        count += 1

    if should_print:
        print("\nAmount Batches:", count, "\nCalculated:", len(train.index)//BATCH_SIZE)
    N_BATCHES = count

    if type(X_test) != type(None):
        return dataloader, test_dataloader, COMMON_BATCH_SHAPE, N_FEATURES, N_BATCHES
    else:
        return dataloader, COMMON_BATCH_SHAPE, N_FEATURES, N_BATCHES



