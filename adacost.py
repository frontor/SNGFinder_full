# -*- coding: utf-8 -*-
"""
adacost.py -- self-contained AdaCost classifier (Fan, Stolfo & Zhang, ICML 1999).

The revision-patch training script (ModelTrain_v3_...py) imports this module as

    from adacost import AdaCostClassifier

AdaCost is a cost-sensitive variant of AdaBoost: the boosting weight update is
multiplied by a cost-adjustment factor beta so that a false negative on the
costly (minority / positive) class is penalised more heavily than a false
positive, and correctly classified examples keep their standard decay.

    beta_i = FNcost   if y_i = 1 and h(x_i) = 0   (false negative)
    beta_i = FPcost   if y_i = 0 and h(x_i) = 1   (false positive)
    beta_i = 1        otherwise

FNcost = "auto" sets FNcost = n_negative / n_positive, i.e. the class-imbalance
ratio, which is the usual default.

Why this file exists
--------------------
`adacost` is not distributed on PyPI, so the original top-level
`from adacost import AdaCostClassifier` made the AdaCost branch of the
submitted script unrunnable for anyone without a private copy. This module
vendors a dependency-free implementation so that both classifier branches
("XGBoost" and "AdaCost") reproduce from this package alone.

Interface follows the scikit-learn estimator convention (explicit __init__
arguments stored unchanged) so the class works inside GridSearchCV,
cross_val_score, TunedThresholdClassifierCV, clone(), etc.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

__all__ = ["AdaCostClassifier"]


def _as_float(value, name):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a positive number, got %r" % (name, value))


class AdaCostClassifier(BaseEstimator, ClassifierMixin):
    """Cost-sensitive AdaBoost (AdaCost) classifier.

    Parameters
    ----------
    base_estimator : object, default=None
        Base learner. Must support ``fit(X, y, sample_weight=...)`` and
        ``predict``/``predict_proba``. ``None`` -> ``DecisionTreeClassifier(max_depth=1)``
        (a decision stump, the conventional AdaBoost base learner).

    n_estimators : int, default=150
        Maximum number of boosting rounds.

    learning_rate : float, default=0.5
        Shrinks the contribution of each round. Trade-off with ``n_estimators``.

    FNcost : float or "auto", default="auto"
        Cost factor applied to false negatives (true class 1 predicted as 0).
        ``"auto"`` -> ``n_negative / n_positive`` computed at fit time.

    FPcost : float, default=1
        Cost factor applied to false positives (true class 0 predicted as 1).

    algorithm : {"SAMME", "SAMME.R"}, default="SAMME"
        ``SAMME`` is the discrete multi-class AdaBoost of Zhu et al. (2009).
        ``SAMME.R`` is the real-valued variant and requires a base estimator
        that provides calibrated ``predict_proba``.

    random_state : int or None, default=None
        Seed for the base estimator.
    """

    def __init__(self, base_estimator=None, n_estimators=150, learning_rate=0.5,
                 FNcost="auto", FPcost=1, algorithm="SAMME", random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.FNcost = FNcost
        self.FPcost = FPcost
        self.algorithm = algorithm
        self.random_state = random_state

    # ------------------------------------------------------------------ utils
    def _make_base(self, random_state):
        if self.base_estimator is not None:
            est = clone(self.base_estimator)
            try:
                est.set_params(random_state=random_state)
            except (ValueError, TypeError):
                pass
            return est
        return DecisionTreeClassifier(max_depth=1, random_state=random_state)

    def _beta(self, y_true, y_pred):
        """AdaCost cost-adjustment factor (Fan et al. 1999)."""
        pos, neg = self.classes_[1], self.classes_[0]
        fn = _as_float(self.FNcost_, "FNcost")
        fp = _as_float(self.FPcost, "FPcost")
        out = np.ones(y_true.shape[0], dtype=float)
        out[(y_true == pos) & (y_pred == neg)] = fn   # false negative
        out[(y_true == neg) & (y_pred == pos)] = fp   # false positive
        return out

    # -------------------------------------------------------------------- fit
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-D, got shape %r" % (X.shape,))
        self.classes_ = np.unique(y)
        if self.classes_.shape[0] != 2:
            raise ValueError("AdaCostClassifier supports binary classification only; "
                             "got classes %r" % (self.classes_,))
        self.n_classes_ = 2

        n_samples = X.shape[0]
        n_pos = int(np.sum(y == self.classes_[1]))
        n_neg = n_samples - n_pos
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Both classes must be present in y.")
        self.FNcost_ = (float(n_neg) / float(n_pos)) if self.FNcost == "auto" else self.FNcost
        _as_float(self.FNcost_, "FNcost")
        _as_float(self.FPcost, "FPcost")
        if self.FNcost_ <= 0 or _as_float(self.FPcost, "FPcost") <= 0:
            raise ValueError("FNcost and FPcost must be positive.")
        if self.algorithm not in ("SAMME", "SAMME.R"):
            raise ValueError("algorithm must be 'SAMME' or 'SAMME.R', got %r" % (self.algorithm,))

        n_estimators = int(self.n_estimators)
        learning_rate = _as_float(self.learning_rate, "learning_rate")
        if n_estimators <= 0 or learning_rate <= 0:
            raise ValueError("n_estimators and learning_rate must be positive.")

        sample_weight = np.full(n_samples, 1.0 / n_samples, dtype=float)
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        rng = np.random.RandomState(self.random_state)

        for iboost in range(n_estimators):
            est = self._make_base(rng.randint(0, np.iinfo(np.int32).max))
            est.fit(X, y, sample_weight=sample_weight)
            y_pred = est.predict(X)
            incorrect = (y_pred != y).astype(float)
            err = float(np.sum(sample_weight * incorrect) / np.sum(sample_weight))

            if err <= 0.0 or err >= 1.0:
                # Perfect or hopeless round: AdaBoost stops here.
                break

            if self.algorithm == "SAMME":
                alpha = learning_rate * (np.log((1.0 - err) / err)
                                         + np.log(self.n_classes_ - 1.0))
            else:  # SAMME.R -- for the binary case this coincides with SAMME.
                alpha = learning_rate * (self.n_classes_ - 1.0) * (
                    np.log((1.0 - err) / err) + np.log(self.n_classes_ - 1.0)
                )

            if alpha <= 0:
                break

            beta = self._beta(y, y_pred)
            sample_weight = sample_weight * np.exp(alpha * incorrect * beta)
            total = sample_weight.sum()
            if not np.isfinite(total) or total <= 0:
                break
            sample_weight = sample_weight / total

            self.estimators_.append(est)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(err)

        if not self.estimators_:
            # Degenerate fallback: a single unweighted stump keeps the estimator usable.
            est = self._make_base(rng.randint(0, np.iinfo(np.int32).max))
            est.fit(X, y)
            self.estimators_ = [est]
            self.estimator_weights_ = [1.0]
            self.estimator_errors_ = [0.5]

        self.estimator_weights_ = np.asarray(self.estimator_weights_, dtype=float)
        self.n_estimators_ = len(self.estimators_)
        return self

    # ---------------------------------------------------------------- predict
    def decision_function(self, X):
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        scores = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for est, alpha in zip(self.estimators_, self.estimator_weights_):
            pred = est.predict(X)
            for k, cls in enumerate(self.classes_):
                scores[:, k] += alpha * (pred == cls)
        # positive-class score minus negative-class score
        return scores[:, 1] - scores[:, 0]

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        scores = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for est, alpha in zip(self.estimators_, self.estimator_weights_):
            pred = est.predict(X)
            for k, cls in enumerate(self.classes_):
                scores[:, k] += alpha * (pred == cls)
        total = scores.sum(axis=1, keepdims=True)
        # guard against an all-zero row
        total[total == 0] = 1.0
        proba = scores / total
        return proba

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        scores = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for est, alpha in zip(self.estimators_, self.estimator_weights_):
            pred = est.predict(X)
            for k, cls in enumerate(self.classes_):
                scores[:, k] += alpha * (pred == cls)
        return self.classes_[np.argmax(scores, axis=1)]

    @property
    def feature_importances_(self):
        """Sum of base-learner feature importances, weighted by round weight."""
        check_is_fitted(self, "estimators_")
        imp = None
        for est, alpha in zip(self.estimators_, self.estimator_weights_):
            fi = getattr(est, "feature_importances_", None)
            if fi is None:
                return None
            imp = alpha * np.asarray(fi, dtype=float) if imp is None else imp + alpha * fi
        return imp
