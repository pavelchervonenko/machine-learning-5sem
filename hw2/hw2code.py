import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ

    x = np.asarray(feature_vector)
    y = np.asarray(target_vector)
    n = x.shape[0]

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # кандидаты на пороги
    diff = x_sorted[1:] != x_sorted[:-1]
    if not np.any(diff):
        return np.array([]), np.array([]), None, None

    thresholds_all = (x_sorted[1:] + x_sorted[:-1]) / 2.0
    thresholds = thresholds_all[diff]         

    # кумулятивные суммы единиц в отсортированном порядке
    y_sorted = y_sorted.astype(float)
    cum_ones = np.cumsum(y_sorted)
    total_ones = cum_ones[-1]

    # позиции сплитов
    positions = np.nonzero(diff)[0]           
    left_counts = positions + 1               
    right_counts = n - left_counts    

    left_ones = cum_ones[positions]           # число единиц слева
    right_ones = total_ones - left_ones       # число единиц справа

    # доли классов и энтропия Джини в поддеревьях
    p1_left = left_ones / left_counts
    p0_left = 1.0 - p1_left
    H_left = 1.0 - p1_left**2 - p0_left**2

    p1_right = right_ones / right_counts
    p0_right = 1.0 - p1_right
    H_right = 1.0 - p1_right**2 - p0_right**2

    # критерий Q(R)
    n_float = float(n)
    ginis = - (left_counts / n_float) * H_left - (right_counts / n_float) * H_right

    # выбираем лучший порог
    best_idx = int(np.argmax(ginis))
    threshold_best = float(thresholds[best_idx])
    gini_best = float(ginis[best_idx])

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        n_objects, n_features = sub_X.shape

        # ограничение по глубине
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        # ограничение по числу объектов в узле
        if self._min_samples_split is not None and n_objects < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best = None
        threshold_best = None
        categories_split_best = None
        gini_best = None
        split_best = None

        # перебираем признаки и ищем лучший сплит
        for feature in range(n_features):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature].astype(float)

            elif feature_type == "categorical":
                col = sub_X[:, feature]
                counts = Counter(col)
                clicks = Counter(col[sub_y == 1])

                ratios = {cat: (clicks[cat] / counts[cat]) for cat in counts}

                sorted_cats = [cat for cat, _ in sorted(ratios.items(), key=lambda t: t[1])]
                categories_map = {cat: idx for idx, cat in enumerate(sorted_cats)}

                feature_vector = np.array([categories_map[val] for val in col], dtype=float)
            else:
                raise ValueError("Unknown feature type")

            if np.unique(feature_vector).size <= 1:
                continue

            # ищем лучший порог по этому признаку
            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if threshold is None or gini is None:
                continue

            # кандидат на сплит
            split = feature_vector < threshold
            left_count = np.sum(split)
            right_count = n_objects - left_count

            # ограничение по min_samples_leaf
            if self._min_samples_leaf is not None:
                if left_count < self._min_samples_leaf or right_count < self._min_samples_leaf:
                    continue

            # обновляем лучший сплит
            if gini_best is None or gini > gini_best:
                gini_best = gini
                feature_best = feature
                split_best = split

                if feature_type == "real":
                    threshold_best = threshold
                    categories_split_best = None
                else:
                    categories_split_best = [
                        cat for cat, idx in categories_map.items() if idx < threshold
                    ]
                    threshold_best = None

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        else:
            node["categories_split"] = categories_split_best

        node["left_child"], node["right_child"] = {}, {}

        # рекурсивно строим поддеревья
        self._fit_node(sub_X[split_best], sub_y[split_best], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split_best], sub_y[~split_best], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "real":
            threshold = node["threshold"]
            if x[feature] < threshold:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        elif feature_type == "categorical":
            categories_left = node["categories_split"]
            if x[feature] in categories_left:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Unknown feature type")

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
