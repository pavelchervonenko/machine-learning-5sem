# Лабораторная работа 2

Кредитный скоринг: ансамблевые методы, собственные реализации бэггинга и
градиентного бустинга, сравнение библиотек бустинга и настройка гиперпараметров
с помощью Optuna. Задача — предсказание `LoanApproved` по персональным и
финансовым признакам (Kaggle-соревнование *mai-ml-lab-2-fiit-2025*).

---

#### **lab2.ipynb**
- исследовательский анализ данных (EDA):
  - распределение целевой переменной `LoanApproved`;
  - анализ признаков (доходы, долги, кредитная история, цели кредита);
  - матрица корреляций числовых признаков;
- подготовка данных:
  - разбор `ApplicationDate` → `ApplicationYear`, `ApplicationMonth`, `ApplicationDayOfWeek`;
  - обработка пропусков и кодирование категориальных признаков (OneHotEncoder);
  - отдельные препроцессоры для логистической регрессии, деревьев и бустингов;
- собственные реализации метрик:
  - `Accuracy`, `Precision`, `Recall`, `F1-score`;
  - `ROC-AUC`, `PR-AUC (Average Precision)` и сравнение со `sklearn.metrics`;
- базовые модели:
  - `LogisticRegression` (baseline);
  - `RandomForestClassifier`;
- собственный бэггинг:
  - класс `CustomBaggingClassifier` (bootstrap + усреднение вероятностей);
  - сравнение с `sklearn.ensemble.BaggingClassifier`;
- собственный градиентный бустинг:
  - класс `CustomGradientBoostingClassifier` (логистическая потеря, деревья-регрессоры);
  - сравнение с `sklearn.ensemble.GradientBoostingClassifier`;
- сравнение реализаций градиентного бустинга:
  - `GradientBoostingClassifier` (sklearn);
  - `XGBClassifier` (XGBoost);
  - `LGBMClassifier` (LightGBM);
  - `CatBoostClassifier` (CatBoost);
- подбор гиперпараметров:
  - настройка `LGBMClassifier` с помощью `Optuna` по метрике ROC-AUC;
- финальная модель и Kaggle submission:
  - обучение LightGBM на всей обучающей выборке с подобранными гиперпараметрами;
  - предсказание вероятностей `P(LoanApproved = 1)` для тестового набора;
  - формирование файла `submission.csv` для отправки на Kaggle.

#### **datasets**
- `train_c.csv` — обучающая выборка (с колонкой `LoanApproved`);
- `test_c.csv` — тестовая выборка (с колонкой `ID`);
- `submission.csv` — финальный файл для Kaggle.

#### **служебные файлы**
- `catboost_info/` — служебная директория, автоматически создаваемая CatBoost.
  Используется для логов обучения и технической информации, в коде напрямую
  не используется.

> Основная метрика соревнования — ROC-AUC. В ноутбуке достигается значение
> ROC-AUC на валидации значительно выше порогового уровня 0.75, необходимого
> для допуска к защите.
