# Домашнее задание 2


#### **h2-trees.ipynb**
- эксперименты с решающими деревьями (`sklearn` и своё);
- визуализация разделяющих поверхностей;
- подбор `max_depth`, `min_samples_split`, `min_samples_leaf`;
- сравнение качества на датасетах UCI.

#### **h2code.py**
- реализация `find_best_split` (критерий Джини);
- класс `DecisionTree` для вещественных и категориальных признаков;
- поддержка `max_depth`, `min_samples_split`, `min_samples_leaf`.

### **datasets**
- `students.csv` — датасет User Knowledge;
- `agaricus-lepiota.data` — грибы (mushrooms);
- `tic-tac-toe-endgame.csv` — крестики-нолики.

> `car.data` и `nursery.data` подгружаются из UCI по URL.