import copy
import torch
import numpy as np


def monte_carlo_dropout_inference(model, dataloader, device, total_classes, num_samples=50):
    model.train()
    predictions = []

    with torch.no_grad():
        for _, inputs, _ in dataloader:
            inputs = inputs.to(device)

            batch_size = inputs.size(0)
            outputs = torch.zeros((num_samples, batch_size, total_classes), device=device)

            for i in range(num_samples):

                outputs[i] = model(inputs)["logits"]

            softmax_outputs = outputs.softmax(dim=-1)

            predictions.append(softmax_outputs.cpu().numpy())

    predictions = np.concatenate(predictions, axis=1)
    return predictions


def compute_distance_and_uncertainty(predictions, samples_per_task):
    mean_predictions = torch.mean(torch.tensor(predictions), dim=0)
    var_predictions = torch.var(torch.tensor(predictions), dim=0)
    mean_predictions = mean_predictions.numpy()
    var_predictions = var_predictions.numpy()
    sorted_rows = np.sort(mean_predictions, axis=1)

    max_values = sorted_rows[:, -1]
    sec_values = sorted_rows[:, -2]

    distance = max_values - sec_values
    distance = distance[:samples_per_task]

    sorted_indices = np.argsort(mean_predictions, axis=1)
    max_indices = sorted_indices[:, -1]
    sec_indices = sorted_indices[:, -2]
    var_1 = var_predictions[np.arange(var_predictions.shape[0]), max_indices]
    var_2 = var_predictions[np.arange(var_predictions.shape[0]), sec_indices]
    uncertainty = var_1 + var_2
    uncertainty = uncertainty[:samples_per_task]

    corr_matrix = np.corrcoef(distance, uncertainty)
    return distance, uncertainty, corr_matrix

def get_prediction(model, dataloader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for _, inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)["logits"]
            softmax_outputs = outputs.softmax(dim=-1)
            predictions.append(softmax_outputs.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    return predictions

def compute_uncertainty(predictions):
    predictions = torch.tensor(predictions)
    entropy = -torch.sum(predictions * torch.log2(predictions + 1e-10), dim=1)
    entropy = entropy.numpy()
    return entropy

def copy_and_dropout(model):
    model_copy = copy.deepcopy(model)
    model_copy.is_dropout = False
    return model_copy