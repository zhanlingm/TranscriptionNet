import time
import copy
import torch
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from config_parser import Device
from PMSELoss import PMSELoss


def train_function(train_model, train_dataloader, optimizer, beta):
    """
    FunDNN model training function.

    Args:
        train_model : FunDNN initialization model.
        train_dataloader : Training set batch data.
        optimizer : Adadelta optimizer.
        beta (float): Weight hyperparameter of the combination of mse and pcc(see Class TranscriptionNet_Hyperparameters).

    Returns:
        Float: Combination loss, mse loss and pearson_loss of train sets.
    """

    train_loss = 0
    mse_loss = 0
    pcc_loss = 0
    num_batches = len(train_dataloader)

    train_model.train()
    for node_feature, gecs_data in train_dataloader:
        node_feature = node_feature.to(Device())
        gecs_data = gecs_data.to(Device())

        optimizer.zero_grad()
        predict_gecs = train_model(node_feature)
        loss, mse, pcc = PMSELoss(gecs_data, predict_gecs, beta)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item()
            mse_loss += mse.item()
            pcc_loss += pcc.item()

    train_loss /= num_batches
    mse_loss /= num_batches
    pcc_loss /= num_batches
    return train_loss, mse_loss, pcc_loss


def valid_function(eval_model, valid_dataloader, beta):
    """
    FunDNN model validation function.

    Args:
        eval_model : The FunDNN model after training on the training set.
        valid_dataloader : Validation set batch data.
        beta (float): Weight hyperparameter of the combination of mse and pcc(see Class TranscriptionNet_Hyperparameters).

    Returns:
        Float: Combination loss, mse loss and pearson_loss of valid sets.
    """

    valid_loss = 0
    mse_loss = 0
    pcc_loss = 0
    num_batches = len(valid_dataloader)

    eval_model.eval()
    with torch.no_grad():
        for node_feature, gecs_data in valid_dataloader:
            node_feature = node_feature.to(Device())
            gecs_data = gecs_data.to(Device())

            predict_gecs = eval_model(node_feature)
            loss, mse, pcc = PMSELoss(gecs_data, predict_gecs, beta)
            valid_loss += loss.item()
            mse_loss += mse.item()
            pcc_loss += pcc.item()

    valid_loss /= num_batches
    mse_loss /= num_batches
    pcc_loss /= num_batches
    return valid_loss, mse_loss, pcc_loss


def test_evaluate(test_model, feature_test, gecs_test):
    """
    FunDNN model test evaluation function.

    Args:
        test_model : The FunDNN model after all iterations of training.
        feature_test (tensor): Test set of node features.
        gecs_test (ndarray): Test set of GECs data.

    Returns:
        None
    """

    feature_test = feature_test.to(Device())
    feature_test_predict = test_model(feature_test).cpu().detach().numpy()

    # feature_test_predict_df = pd.DataFrame(feature_test_predict, index=net_test.index)
    # feature_test_predict_df.to_csv(save_path + "feature_test_predict.csv", index=True)

    # 计算Pearson相关性
    pcc = []
    for i in range(feature_test_predict.shape[0]):
        pearson = np.corrcoef(feature_test_predict[i], gecs_test[i])[0, 1]
        pcc.append(pearson)
    abs_pcc = abs(np.array(pcc))
    abs_pcc_mean = abs_pcc.mean()

    # 计算均方误差
    mse = mean_squared_error(gecs_test, feature_test_predict)

    # 计算ks_test
    gecs_test_flaten = np.ravel(gecs_test)
    feature_test_predict_flaten = np.ravel(feature_test_predict)
    d, p_value = stats.ks_2samp(gecs_test_flaten, feature_test_predict_flaten)

    print('=' * 30)
    print('test evaluate result:\nAverage pcc: {}\nAverage mse: {}\nAverage D: {}'
          .format(abs_pcc_mean, mse, d))
    print('=' * 30)

    # return abs_pearson


def plot_loss_figure(epochs, train_loss, valid_loss):
    """
    Draw the training loss value image

    Args:
        epochs (int): Number of epochs for FunDNN model training
        train_loss (list): List of training set loss values.
        valid_loss (list): List of valid set loss values.
    """
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['legend.fontsize'] = 22
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(7, 7), dpi=144)

    plt.title('FunDNN loss figures')
    plt.plot(range(epochs), train_loss, color='red', linestyle='--', label='train loss', linewidth=2)
    plt.plot(range(epochs), valid_loss, color='dodgerblue', linestyle='-', label='valid loss', linewidth=2)
    plt.legend(loc='upper right', frameon=False)
    plt.xlabel('Epochs')
    plt.ylabel('Loss values')
    plt.show()


def train(epochs, model, train_dataloader, valid_dataloader, optimizer, beta):
    """
    Train the FunDNN model.

    Args:
        epochs (int): Number of epochs for FunDNN model training
        model (nn.Module): FunDNN model
        train_dataloader : Training set batch data.
        valid_dataloader : Validation set batch data.
        optimizer : Adadelta optimizer.
        beta (float): Weight hyperparameter of the combination of mse and pcc(see Class TranscriptionNet_Hyperparameters).
    Returns:
        best_model (nn.Module): The trained FunDNN model with the lowest validation loss.
    """

    min_loss = float("inf")
    best_model = None

    train_losses = []
    valid_losses = []
    for epoch in range(epochs):
        epoch_start_time = time.time()

        tra_loss, tra_mse_loss, tra_cor_loss = train_function(train_model=model,
                                                              train_dataloader=train_dataloader,
                                                              optimizer=optimizer,
                                                              beta=beta)
        val_loss, val_mse_loss, val_cor_loss = valid_function(eval_model=model,
                                                              valid_dataloader=valid_dataloader,
                                                              beta=beta)
        train_losses.append(tra_loss)
        valid_losses.append(val_loss)

        if val_loss < min_loss:
            min_loss = val_loss
            best_model = copy.deepcopy(model)

        print('end of epoch:{:3d} | time:{:5.2f}s | train loss:{:5.5f} | valid loss:{:5.5f} | train MseLoss:{:5.5f} | '
              'train PccLoss:{:5.5f} | valid MseLoss:{:5.5f} | valid PccLoss:{:5.5f}'
              .format(epoch, (time.time() - epoch_start_time), tra_loss, val_loss, tra_mse_loss, tra_cor_loss,
                      val_mse_loss, val_cor_loss))

    # plot_loss_figure(epochs, train_losses, valid_losses)

    return best_model


def feature_predict(best_model, node_feature, save_path, name):
    """
    Predict the pre-GECs

    Args:
        best_model (nn.Module): The trained FunDNN model with the lowest validation loss.
        node_feature: All network nodes embedded features.
        save_path: Path to save the trained model.
        name: GECs type(RNAi, OE or CRISPR)

    Returns:
        pre_GECs (dataframe): pre-GECs.
    """
    node_feature_tensor = torch.FloatTensor(node_feature.values)
    node_feature_tensor = node_feature_tensor.to(Device())

    node_feature_predict = best_model(node_feature_tensor).cpu().detach().numpy()

    node_feature_predict = pd.DataFrame(node_feature_predict, index=node_feature.index).sort_index()
    node_feature_predict.to_csv(save_path + name + "_pre_GECs.csv", index=True, sep=",")

    print('\npredict finish:\npre_GECs size:{}\n'.format(node_feature_predict.shape))
    return node_feature_predict
