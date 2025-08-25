import random
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import re
from scipy.spatial.distance import pdist, euclidean
from scipy.stats import entropy
from collections import Counter

plt.rcParams["font.sans-serif"] = ["Times New Roman"] + plt.rcParams["font.sans-serif"]

def plot_selected_numbers_positions(uncertainty_sort, selected_numbers):
    uncertainty_sort_list = uncertainty_sort.tolist()
    selected_list = selected_numbers.tolist()

    positions = [uncertainty_sort_list.index(num) for num in selected_list]

    xticks = np.arange(0, 5001, 1000)
    plt.xticks(xticks)
    yticks = np.arange(0, 5001, 1000)
    plt.yticks(yticks)

    plt.scatter(selected_numbers, positions)
    plt.xlabel('Indices of selected samples')
    plt.ylabel('Positions in the sequence of ascending order of uncertainty')
    plt.title('Uncertainty of PUC sample selection')
    plt.grid(True)

    y_min = -200
    y_max = 5200
    plt.ylim(y_min, y_max)

    plt.savefig('uncertainty.pdf', format='pdf')
    plt.show()

def merge_lines_to_ndarray(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 10):
        sub_data = []
        for j in range(10):
            if i + j < len(lines):
                line = lines[i + j].strip()
                if line:
                    start_index = line.find('[')
                    if start_index != -1:
                        line = line[start_index + 1:]
                    line_data = list(map(int, line[:-1].split(',')))
                    sub_data.extend(line_data)
        if sub_data:
            sub_ndarray = np.array(sub_data)
            data.append(sub_ndarray)

    return data

def plot_scatter(x, y, x_label='x values', y_label='y values', title='Scatter Plot of x vs y'):

    if x.shape != y.shape:
        raise ValueError("输入的x和y数组维度必须相同")

    plt.scatter(x, y)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.show()

def correlation_analysis(x, y):
    pearson_corr, pearson_p = pearsonr(x, y)
    print(f"pearson_corr: {pearson_corr}, p 值: {pearson_p}")

    spearman_corr, spearman_p = spearmanr(x, y)
    print(f"spearman_corr: {spearman_corr}, p 值: {spearman_p}")

    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('correlation_analysis')
    plt.show()

def save_array(x, prefix):
    storage_folder = 'storage'
    if not os.path.exists(storage_folder):
        os.makedirs(storage_folder)

    file_name = f"{prefix}_sorted_indices.npy"
    file_path = os.path.join(storage_folder, file_name)

    np.save(file_path, x)

def load_array(prefix):
    storage_folder = 'storage'
    file_name = f"{prefix}_sorted_indices.npy"
    file_path = os.path.join(storage_folder, file_name)
    x = np.load(file_path)
    return x

def process_file(file_path):
    processed_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip().startswith('self._selected_indices'):
                continue
            parts = line.strip().split(':')
            x = int(parts[0])
            num_list = list(map(int, re.findall(r'\d+', parts[1])))

            multiple = (x % 10) * 500

            new_num_list = [num + multiple for num in num_list]

            new_line = f"{x}: {new_num_list}\n"
            processed_lines.append(new_line)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(processed_lines)
        print("written")

def catch_impact_of_decision_layer(features, predictions):
    pass

def calculate_pairwise_distances(features):
    distances = pdist(features, metric='euclidean')
    distances = (distances.T / (np.linalg.norm(distances.T, axis=0) + 1e-8)).T
    return distances

def calculate_pairwise_distances_split(features, m):
    # m = m / 2
    features_1 = features[0:m, :]
    features_2 = features[m:2*m, :]
    distances = []
    for i in range(m):
        for j in range(m):
            dist = euclidean(features_1[i], features_2[j])
            distances.append(dist)
    return distances

def calculate_pairwise_kl_divergence(predictions):
    num_samples = predictions.shape[0]
    kl_divergences = []
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            kl = 0.5 * (entropy(predictions[i], predictions[j]) + entropy(predictions[j], predictions[i]))
            kl_divergences.append(kl)
    kl_divergences = np.array(kl_divergences)
    total_sum = np.sum(kl_divergences)
    avg_kl = total_sum / ((num_samples * (num_samples-1)) /2)
    print("avg_KL:", avg_kl)
    return kl_divergences

def calculate_pairwise_kl_divergence_new(predictions, t):
    num_samples = predictions.shape[0]
    kl_divergences, idx = [], []
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            kl = 0.5 * (entropy(predictions[i], predictions[j]) + entropy(predictions[j], predictions[i]))
            if kl >= t:
                idx.append(i)
                idx.append(j)
            kl_divergences.append(kl)
    kl_divergences = np.array(kl_divergences)
    total_sum = np.sum(kl_divergences)
    avg_kl = total_sum / ((num_samples * (num_samples-1)) / 2)
    counter = Counter(idx)

    sorted_items = sorted(counter.items(), key=lambda item: item[1])
    sorted_numbers = [item[0] for item in sorted_items]
    sorted_counts = [item[1] for item in sorted_items]
    return kl_divergences, sorted_numbers, sorted_counts, len(idx), sorted_numbers, sorted_counts

def calculate_pairwise_kl_divergence_split(predictions, m):
    # m = m/2
    predictions_1 = predictions[0:m, :]
    predictions_2 = predictions[m:2*m, :]
    kl_divergences = []
    for i in range(m):
        for j in range(m):
            kl = entropy(predictions_1[i], predictions_2[j])
            kl_divergences.append(kl)
    return np.array(kl_divergences)


def catch_relation(features, predictions, t):
    normalized_distances = calculate_pairwise_distances(features)
    kl_divergences, _, _, length, sorted_numbers, sorted_counts = calculate_pairwise_kl_divergence_new(predictions,t)

    return length, sorted_numbers, sorted_counts

def plot_relation(features, predictions, cur_task=None):
    distances = calculate_pairwise_distances(features)
    kl_divergences = calculate_pairwise_kl_divergence(predictions)
    filename = os.path.join("Figures", f"{cur_task}_plot_relation.pdf")

    plt.scatter(distances, kl_divergences, s=10, c='b', alpha=0.5)
    plt.xlabel('L2 Distance of Features')
    plt.ylabel('KL Divergence of Predictions')
    plt.title('Relationship between Feature Distance and KL Divergence')
    plt.savefig(filename)
    plt.close()

def catch_relation_split(features, predictions, m, cur_task=None):
    distances = calculate_pairwise_distances_split(features, m)
    kl_divergences = calculate_pairwise_kl_divergence_split(predictions, m)
    filename = os.path.join("Figures", f"{cur_task}_catch_relation_split.pdf")

    plt.scatter(distances, kl_divergences, c="b",s=10, alpha=0.5)
    plt.xlabel('L2 Distance of Features')
    plt.ylabel('KL Divergence of Predictions')
    plt.title('Relationship between Feature Distance and KL Divergence')
    plt.savefig(filename)
    plt.close()

def color_contrast(features, predictions, lst, cur_task=None, i=None):
    distances = calculate_pairwise_distances(features)
    kl_divergences = calculate_pairwise_kl_divergence(predictions)
    filename = os.path.join("Figures", f"{cur_task,i}_catch_color_contrast.pdf")

    idx = find_index(len(features), lst)
    colors = ["b"] * len(distances)
    for i in idx:
        colors[i] = "r"

    plt.scatter(distances, kl_divergences, s=10, c=colors,alpha=0.5)
    plt.xlabel('L2 Distance of Features')
    plt.ylabel('KL Divergence of Predictions')
    plt.title('Relationship between Feature Distance and KL Divergence')
    plt.savefig(filename)
    plt.close()

def find_index_k(num_of_data, k):
    index_list = []
    n = num_of_data
    for i in range(k):
        index = (i * (2 * n - i - 1) // 2) + (k - i - 1)
        index_list.append(index)
    for j in range(k + 1, n):
        index = (k * (2 * n - k - 1) // 2) + (j - k - 1)
        index_list.append(index)
    print(f"涉及第 {k} 个点的距离的索引: {sorted(index_list)}")
    return index_list

def find_index(num_of_data,l):
    lst, s = [], []
    for i in l:
        lst.append(find_index_k(num_of_data, i))
    for j in range(len(lst)):
        s += lst[j]
    s = set(s)
    s = list(s)
    return np.array(s)

def color_contrast_new(features, predictions, t):
    distances = calculate_pairwise_distances(features)
    (kl_divergences, sorted_numbers, sorted_counts, length, sorted_numbers,
     sorted_counts) = calculate_pairwise_kl_divergence_new(predictions, t)

    print("idx:", sorted_numbers, "len:",len(distances))
    colors = ["b"] * len(distances)
    for i in sorted_numbers:
        colors[i] = "r"

    plt.scatter(distances, kl_divergences, s=10, c=colors,alpha=0.5)

    plt.ylim(-0.1, 2.0)
    plt.yticks(np.arange(0, 2.0, 0.2))

    plt.xlabel('L2 Distance of Features')
    plt.ylabel('KL Divergence of Predictions')
    plt.title('Relationship between Feature Distance and KL Divergence')
    plt.show()



