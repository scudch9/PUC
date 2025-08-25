import copy
import logging
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import os
from analysis_tools import *
from uncertainty_tools import get_prediction

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self._fake_dm, self.fake_tm = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

        self._uncertainty = None
        self._memory_uncertainty = None
        self._selected_indices = {}

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            if self._cur_task <= -1:
                self._reduce_exemplar(data_manager, per_class)
                self._construct_exemplar(data_manager, per_class)
            else:
                self._reduce_exemplar(data_manager, per_class)
                self._construct_exemplar_custom2(data_manager, per_class)


    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self, save_conf=False):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        if save_conf:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
            _target_path = os.path.join(self.args['logfilename'], "target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

            _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
            os.makedirs(_save_dir, exist_ok=True)
            _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
            with open(_save_path, "a+") as f:
                f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")

        return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _get_fake_m(self):
        return (self._fake_dm, self._fake_tm)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]

            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference
                # selected_indices.append(original_indices[i])  # 记录选取的索引

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection
                # original_indices = np.delete(original_indices, i)  # 删除已选索引


            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

        if self._cur_task >= 1:
            md = data_manager.get_memory_dataset(
                mode="train",
                appendent=self._get_memory(),
            )
            memory_loader = DataLoader(
                md, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors_, _ = self._extract_vectors(memory_loader)
            predictions_ = get_prediction(self._network, memory_loader, self._device)
            vectors_0 = vectors_[:m,:]
            vectors_1 = vectors_[m:2*m,:]
            vectors__ = np.concatenate((vectors_0, vectors_1))
            predictions_0 = predictions_[:m,:]
            predictions_1 = predictions_[m:2*m,:]
            predictions__ = np.concatenate((predictions_0, predictions_1))
            catch_relation_split(vectors__ , predictions__, m, self._cur_task)

            if self._cur_task >= 1 and self._cur_task <= 4:
                for i in range(10):
                    vectors___ = vectors_[m * self._cur_task * 10 + i * m: m * self._cur_task * 10 + m * (i+1),:]
                    predictions___ = predictions_[m * self._cur_task * 10 + i * m: m * self._cur_task * 10 + m * (i+1),:]
                    plot_relation(vectors___, predictions___, self._cur_task)
                    idx_uncertainty = self._memory_uncertainty[m * self._cur_task * 10 + i * m: m * self._cur_task * 10 + m * (i+1)]
                    idx_sorted_uncertainty = np.argsort(idx_uncertainty)
                    lst = idx_sorted_uncertainty[-int(m*0.25):]
                    color_contrast(vectors___, predictions___, lst, self._cur_task, i)


    def _construct_exemplar_new(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]

            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference
                # selected_indices.append(original_indices[i])  # 记录选取的索引

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            fake_selected_exemplars = np.array(selected_exemplars)
            fake_exemplar_targets = np.full(m, class_idx)

            self._fake_dm = (
                np.concatenate((self._data_memory, fake_selected_exemplars))
                if len(self._data_memory) != 0
                else fake_selected_exemplars
            )
            self._fake_tm = (
                np.concatenate((self._targets_memory, fake_exemplar_targets))
                if len(self._targets_memory) != 0
                else fake_exemplar_targets
            )
            length, sorted_numbers, sorted_counts, vec, predictions = self._catch_fake_relation(
                data_manager, self.samples_per_class, class_idx
            )
            selected_exemplars, exemplar_vectors, num_of_drift_samples, drift_index = self._remove_drift_sample(
                fake_selected_exemplars, exemplar_vectors, length, sorted_numbers, sorted_counts, self.samples_per_class
            )
            color_contrast(vec, predictions, drift_index)

            for k in range(m + 1 - num_of_drift_samples, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference
                # selected_indices.append(original_indices[i])  # 记录选取的索引

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )

            selected_exemplars = np.array(selected_exemplars, dtype=np.uint8)
            exemplar_targets = np.full(m, class_idx)

            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            vec, predictions = self._catch_relation(
                data_manager, self.samples_per_class, class_idx
            )
            plot_relation(vec, predictions)

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_re(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]

            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference
                # selected_indices.append(original_indices[i])  # 记录选取的索引

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            fake_selected_exemplars = np.array(selected_exemplars)
            fake_exemplar_targets = np.full(m, class_idx)

            self._fake_dm = (
                np.concatenate((self._data_memory, fake_selected_exemplars))
                if len(self._data_memory) != 0
                else fake_selected_exemplars
            )
            self._fake_tm = (
                np.concatenate((self._targets_memory, fake_exemplar_targets))
                if len(self._targets_memory) != 0
                else fake_exemplar_targets
            )

            length, sorted_numbers, sorted_counts, vec, predictions = self._catch_fake_relation(
                data_manager, self.samples_per_class, class_idx
            )
            selected_exemplars, exemplar_vectors, num_of_drift_samples, drift_index = self._remove_drift_sample(
                fake_selected_exemplars, exemplar_vectors, length, sorted_numbers, sorted_counts, self.samples_per_class
            )
            color_contrast(vec, predictions, drift_index)

            selected_exemplars = selected_exemplars + selected_exemplars[:num_of_drift_samples]

            selected_exemplars = np.array(selected_exemplars, dtype=np.uint8)
            exemplar_targets = np.full(m, class_idx)

            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            vec, predictions = self._catch_relation(
                data_manager, self.samples_per_class, class_idx
            )
            plot_relation(vec, predictions)

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    @staticmethod
    def _remove_drift_sample(
            selected_exemplar, exemplar_vectors, length, sorted_numbers, sorted_counts, samples_per_class
    ):
        n = len(sorted_counts)
        num_of_drift_samples = 0
        for m in range(1, n + 1):
            # 计算列表最后 m 个数的和
            last_m_sum = sum(sorted_counts[-m:])
            if last_m_sum > length * 0.4:
                num_of_drift_samples = m
                break
        drift_idx = sorted_numbers[-num_of_drift_samples:]
        drift_idx_ = [i - samples_per_class for i in drift_idx]
        mask = np.ones(len(selected_exemplar), dtype=bool)
        mask[drift_idx_] = False
        selected_exemplar = selected_exemplar[mask]
        exemplar_vectors = np.array(exemplar_vectors)
        exemplar_vectors = exemplar_vectors[mask]
        return selected_exemplar.tolist(), exemplar_vectors.tolist(), num_of_drift_samples, drift_idx

    def _catch_fake_relation(self, data_manager, m, class_idx, t=0.3):
        fake_md = data_manager.get_memory_dataset(
            mode="train",
            appendent=self._get_fake_m(),
        )
        memory_loader = DataLoader(
            fake_md, batch_size=batch_size, shuffle=False, num_workers=4
        )
        vectors, _ = self._extract_vectors(memory_loader)
        predictions = get_prediction(self._network, memory_loader, self._device)
        vectors, predictions = vectors[class_idx * m : (class_idx + 1) * m, :], predictions[class_idx * m : (class_idx + 1) * m, :]
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        # catch_relation(vectors, predictions)
        length, sorted_numbers, sorted_counts = catch_relation(vectors, predictions, t)
        # color_contrast(vectors, predictions, self._memory_size, top_5_indices)
        return length, sorted_numbers, sorted_counts, vectors, predictions

    def _catch_relation(self, data_manager, m, class_idx):
        md = data_manager.get_memory_dataset(
            mode="train",
            appendent=self._get_memory(),
        )
        memory_loader = DataLoader(
            md, batch_size=batch_size, shuffle=False, num_workers=4
        )
        vectors, _ = self._extract_vectors(memory_loader)
        predictions = get_prediction(self._network, memory_loader, self._device)
        vectors, predictions = vectors[class_idx * m : (class_idx + 1) * m, :], predictions[class_idx * m : (class_idx + 1) * m, :]
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        return vectors, predictions

    def _construct_exemplar_custom(self, data_manager, m):
        m_reduced = int(m * 0.8)
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            # print('data:',data, 'shape', data.shape) #(500,32,32,3)
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m_reduced + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m_reduced, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

        if self._cur_task <= 1:
            self._lu_exemplar(data_manager, self._memory_size - len(self._targets_memory))
        else:
            self._hu_exemplar(data_manager, self._memory_size - len(self._targets_memory))

    def _construct_exemplar_custom2(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        selected_indi = np.array([])
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            sorted_indices = np.argsort(self._uncertainty)

            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            k = 1
            while k <= m:
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                index = i + 500 * (class_idx % 10)
                index_in_sorted_indices = np.where(sorted_indices == index)[0][0]
                # print("index_in_sorted_indices:",index_in_sorted_indices)

                if index_in_sorted_indices >= self.args["beta"] - self._cur_task * self.args["alpha"] :
                    vectors = np.delete(
                        vectors, i, axis=0
                    )  # Remove it to avoid duplicative selection
                    data = np.delete(
                        data, i, axis=0
                    )  # Remove it to avoid duplicative selection
                    k -= 1
                else:
                    selected_indi = np.append(selected_indi, index)
                    selected_exemplars.append(
                        np.array(data[i])
                    )  # New object to avoid passing by inference
                    exemplar_vectors.append(
                        np.array(vectors[i])
                    )  # New object to avoid passing by inference

                    vectors = np.delete(
                        vectors, i, axis=0
                    )  # Remove it to avoid duplicative selection
                    data = np.delete(
                        data, i, axis=0
                    )  # Remove it to avoid duplicative selection
                k += 1

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_combined(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        selected_indi = np.array([])
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            # print('data:',data, 'shape', data.shape) #(500,32,32,3)
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            sorted_indices = np.argsort(self._uncertainty)
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            k = 1
            while k <= m:
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                index = i + 500 * (class_idx % 10)
                index_in_sorted_indices = np.where(sorted_indices == index)[0][0]
                if self._cur_task <= 10:
                    if index_in_sorted_indices >= 4200:
                        vectors = np.delete(
                            vectors, i, axis=0
                        )  # Remove it to avoid duplicative selection
                        data = np.delete(
                            data, i, axis=0
                        )  # Remove it to avoid duplicative selection
                        k -= 1
                    else:
                        selected_indi = np.append(selected_indi, index)
                        selected_exemplars.append(
                            np.array(data[i])
                        )  # New object to avoid passing by inference
                        exemplar_vectors.append(
                            np.array(vectors[i])
                        )  # New object to avoid passing by inference

                        vectors = np.delete(
                            vectors, i, axis=0
                        )  # Remove it to avoid duplicative selection
                        data = np.delete(
                            data, i, axis=0
                        )  # Remove it to avoid duplicative selection
                else:
                    if index_in_sorted_indices >= 4500:
                        vectors = np.delete(
                            vectors, i, axis=0
                        )  # Remove it to avoid duplicative selection
                        data = np.delete(
                            data, i, axis=0
                        )  # Remove it to avoid duplicative selection
                        k -= 1
                    else:
                        selected_indi = np.append(selected_indi, index)
                        selected_exemplars.append(
                            np.array(data[i])
                        )  # New object to avoid passing by inference
                        exemplar_vectors.append(
                            np.array(vectors[i])
                        )  # New object to avoid passing by inference

                        vectors = np.delete(
                            vectors, i, axis=0
                        )  # Remove it to avoid duplicative selection
                        data = np.delete(
                            data, i, axis=0
                        )  # Remove it to avoid duplicative selection
                k += 1
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

            fake_selected_exemplars = np.array(selected_exemplars)
            fake_exemplar_targets = np.full(m, class_idx)

            self._fake_dm = (
                np.concatenate((self._data_memory, fake_selected_exemplars))
                if len(self._data_memory) != 0
                else fake_selected_exemplars
            )
            self._fake_tm = (
                np.concatenate((self._targets_memory, fake_exemplar_targets))
                if len(self._targets_memory) != 0
                else fake_exemplar_targets
            )
            length, sorted_numbers, sorted_counts = self._catch_fake_relation(data_manager, self.samples_per_class,
                                                                              class_idx)
            selected_exemplars, exemplar_vectors, num_of_unreliable_samples = self._remove_drift_sample(
                fake_selected_exemplars, exemplar_vectors, length, sorted_numbers, sorted_counts, self.samples_per_class
            )

            for k in range(m + 1 - num_of_unreliable_samples, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference
                # selected_indices.append(original_indices[i])  # 记录选取的索引

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )

            selected_exemplars = np.array(selected_exemplars, dtype=np.uint8)
            exemplar_targets = np.full(m, class_idx)

            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

        # result = merge_lines_to_ndarray("sample.txt")
        # if self._cur_task != 9:
            # plot_selected_numbers_positions(sorted_indices, selected_indi)

    def _hu_exemplar(self, data_manager, m):
        sorted_indices = np.argsort(self._uncertainty)
        top_m_indices = sorted_indices[-m:]
        all_data = []
        all_targets = []
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            all_data.extend(data)
            all_targets.extend(targets)

        all_data = np.array(all_data)
        all_targets = np.array(all_targets)

        selected_exemplars = all_data[top_m_indices]
        exemplar_targets = all_targets[top_m_indices]

        self._data_memory = (
            np.concatenate((self._data_memory, selected_exemplars))
            if len(self._data_memory) != 0
            else selected_exemplars
        )
        self._targets_memory = (
            np.concatenate((self._targets_memory, exemplar_targets))
            if len(self._targets_memory) != 0
            else exemplar_targets
        )

    def _lu_exemplar(self, data_manager, m):
        sorted_indices = np.argsort(self._uncertainty)
        top_m_indices = sorted_indices[:m]
        all_data = []
        all_targets = []
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            all_data.extend(data)
            all_targets.extend(targets)

        all_data = np.array(all_data)
        all_targets = np.array(all_targets)

        selected_exemplars = all_data[top_m_indices]
        exemplar_targets = all_targets[top_m_indices]

        self._data_memory = (
            np.concatenate((self._data_memory, selected_exemplars))
            if len(self._data_memory) != 0
            else selected_exemplars
        )
        self._targets_memory = (
            np.concatenate((self._targets_memory, exemplar_targets))
            if len(self._targets_memory) != 0
            else exemplar_targets
        )

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means
