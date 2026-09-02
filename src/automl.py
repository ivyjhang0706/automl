import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor
from sklearn.svm import SVR
from dl_ml import TabNetRegressorWrapper, ResNetMLPRegressor


class InfToNanImputer(TransformerMixin, BaseEstimator):
    """ECG 特徵有些是比值（分母可能為 0），算出來會是 inf 或 NaN。
    先把 inf 轉成 NaN，再用「只用 train fold 算出的欄位平均值」補值，
    避免污染到後續模型或 StandardScaler（NaN/inf 進去整欄統計量都會壞掉）。"""

    def __init__(self):
        self.imputer_ = SimpleImputer(strategy='mean')

    def fit(self, X, y=None):
        X = np.where(np.isinf(np.asarray(X, dtype=np.float64)), np.nan, X)
        self.imputer_.fit(X)
        return self

    def transform(self, X):
        X = np.where(np.isinf(np.asarray(X, dtype=np.float64)), np.nan, X)
        return self.imputer_.transform(X)


class SimpleAutoMLRegressor:
    """血糖值是連續數值的回歸任務，用 Optuna 在多種回歸器間搜尋，5-fold KFold CV 評分。
    fit() 結束後會用最佳參數在全部訓練資料上重新 refit 一次，存在 self.best_model_，
    供 main.py 在 held-out 的 testdata 上做最終評估。

    n_train 夠大時（> stacking_min_n）會額外嘗試把 Optuna 搜尋過程中，各種 regressor
    表現最好的那組參數組成 StackingRegressor，並用「兩道門」決定要不要採用：

      第一道門（stacking_min_n，粗篩，省算力）：
          n_train 沒超過這個值就直接跳過 stacking，不花額外算力去嘗試。
          小樣本時 stacking 的 out-of-fold meta-feature 也不穩定，試了大概率也白試。

      第二道門（SEM 顯著性檢定，決定要不要採用）：
          n_train 超過門檻、真的算出 stacking 的 CV 分數後，拿它跟單一最佳模型的
          CV 分數比較。用標準誤差（SEM = std / sqrt(fold數)）而不是單純比較平均值，
          避免把 CV 分數的隨機波動誤判成「stacking 真的比較好」：

              stacking_mean - best_mean > k * sqrt(stacking_sem**2 + best_sem**2)

          差距沒有大於雙方變異量的合理範圍，就當作雜訊，維持單一最佳模型。
    """

    def __init__(self, n_trials=30, scoring='neg_mean_absolute_percentage_error',
                 stacking_min_n=600, stacking_significance_k=1.5,
                 max_stacking_estimators=4, cv_n_splits=5):
        self.n_trials = n_trials
        self.scoring = scoring
        self.stacking_min_n = stacking_min_n
        self.stacking_significance_k = stacking_significance_k
        self.max_stacking_estimators = max_stacking_estimators
        self.cv_n_splits = cv_n_splits

        self.best_model_ = None
        self.best_params = None
        self.best_score = -np.inf
        # fit() 結束後記錄本次是否採用 stacking、兩道門各自的判斷依據，供 main.py 落地成 csv 欄位。
        self.ensemble_info_ = None

    def _sample_params(self, trial):
        regressor_name = trial.suggest_categorical(
            'regressor',
            ['RandomForest', 'XGBoost', 'SVR', 'Ridge', 'TabNet', 'ResNetMLP']
        )
        params = {'regressor': regressor_name}

        if regressor_name == 'RandomForest':
            params['rf_max_depth'] = trial.suggest_int('rf_max_depth', 2, 20)
            params['rf_n_estimators'] = trial.suggest_int('rf_n_estimators', 50, 300)

        elif regressor_name == 'XGBoost':
            params['xgb_lr'] = trial.suggest_float('xgb_lr', 1e-3, 0.3, log=True)
            params['xgb_depth'] = trial.suggest_int('xgb_depth', 3, 8)
            params['xgb_alpha'] = trial.suggest_float('xgb_alpha', 1e-3, 10.0, log=True)
            params['xgb_lambda'] = trial.suggest_float('xgb_lambda', 1e-3, 10.0, log=True)
            params['xgb_n_estimators'] = trial.suggest_int('xgb_n_estimators', 50, 300)

        elif regressor_name == 'SVR':
            params['svr_C'] = trial.suggest_float('svr_C', 1e-1, 100.0, log=True) # 2.1.1 Regression c=100
            params['svr_gamma'] = trial.suggest_float('svr_gamma', 1e-3, 10.0, log=True)
            params['svr_epsilon'] = trial.suggest_float('svr_epsilon', 1e-3, 1.0, log=True)

        elif regressor_name == 'TabNet':
            params['tabnet_n_d'] = trial.suggest_int('tabnet_n_d', 8, 32)
            params['tabnet_n_a'] = trial.suggest_int('tabnet_n_a', 8, 32)
            params['tabnet_n_steps'] = trial.suggest_int('tabnet_n_steps', 3, 7)
            params['tabnet_gamma'] = trial.suggest_float('tabnet_gamma', 1.0, 2.0)
            params['tabnet_lr'] = trial.suggest_float('tabnet_lr', 1e-3, 3e-2, log=True)

        elif regressor_name == 'ResNetMLP':
            params['resnet_hidden_dim'] = trial.suggest_categorical('resnet_hidden_dim', [32, 64, 128])
            params['resnet_n_blocks'] = trial.suggest_int('resnet_n_blocks', 2, 3)
            params['resnet_dropout'] = trial.suggest_float('resnet_dropout', 0.0, 0.5)
            params['resnet_lr'] = trial.suggest_float('resnet_lr', 1e-4, 1e-2, log=True)

        else:
            params['ridge_alpha'] = trial.suggest_float('ridge_alpha', 1e-3, 10.0, log=True)

        return params

    def _build_model(self, params):
        regressor_name = params['regressor']

        if regressor_name == 'RandomForest':
            model = RandomForestRegressor(max_depth=params['rf_max_depth'],
                                           n_estimators=params['rf_n_estimators'], random_state=42)

        elif regressor_name == 'XGBoost':
            model = XGBRegressor(learning_rate=params['xgb_lr'],
                                 max_depth=params['xgb_depth'],
                                 reg_alpha=params['xgb_alpha'],
                                 reg_lambda=params['xgb_lambda'],
                                 n_estimators=params['xgb_n_estimators'],
                                 random_state=42)

        elif regressor_name == 'SVR':
            model = SVR(C=params['svr_C'],
                        gamma=params['svr_gamma'],
                        epsilon=params['svr_epsilon'])

        elif regressor_name == 'TabNet':
            model = TabNetRegressorWrapper(n_d=params['tabnet_n_d'], n_a=params['tabnet_n_a'],
                                            n_steps=params['tabnet_n_steps'], gamma=params['tabnet_gamma'],
                                            lr=params['tabnet_lr'], max_epochs=50)

        elif regressor_name == 'ResNetMLP':
            model = ResNetMLPRegressor(hidden_dim=params['resnet_hidden_dim'], n_blocks=params['resnet_n_blocks'],
                                        dropout=params['resnet_dropout'], lr=params['resnet_lr'], epochs=30)

        else:
            model = Ridge(alpha=params['ridge_alpha'], random_state=42)

        # ECG 特徵可能有 inf/NaN（比值特徵分母為 0），每個模型都先過一層 impute。
        # 包成 Pipeline 讓 SimpleImputer 只用每個 CV fold 的 train 子集算平均值，
        # 對 val fold transform，不會有 fold 間洩漏。
        return Pipeline([('impute', InfToNanImputer()), ('model', model)])

    def _cv(self):
        return KFold(n_splits=self.cv_n_splits, shuffle=True, random_state=42)

    @staticmethod
    def _sem(std, n_folds):
        """標準誤差：估計「這個平均分數本身」的不確定性，而不是原始分數的離散度。"""
        return float(std) / np.sqrt(n_folds)

    def _objective(self, trial, X, y):
        params = self._sample_params(trial)
        model = self._build_model(params)

        # K-Fold 交叉驗證（血糖值是連續值，不能用 StratifiedKFold）
        # TabNet/ResNetMLP 共用同一張 GPU，多個 fold 平行跑會搶顯存，因此改成單工執行。
        cv_n_jobs = 1 if params['regressor'] in ('TabNet', 'ResNetMLP') else -1
        scores = cross_val_score(model, X, y, cv=self._cv(), scoring=self.scoring, n_jobs=cv_n_jobs)
        # 把 std 存進 trial，study 跑完後不用重跑就能拿到最佳 trial 的 CV 離散度。
        trial.set_user_attr('cv_std', float(scores.std()))
        return scores.mean()

    def _best_params_per_regressor(self, study):
        """從所有已完成的 trial 裡，取出每種 regressor 表現最好的一組參數，
        當作 stacking 的候選 base estimator（讓 base learner 之間夠多樣，而不是同一種模型的不同超參數）。"""
        best_per_type = {}
        for t in study.trials:
            if t.value is None:
                continue
            reg = t.params.get('regressor')
            if reg is None:
                continue
            if reg not in best_per_type or t.value > best_per_type[reg].value:
                best_per_type[reg] = t

        ranked = sorted(best_per_type.values(), key=lambda t: t.value, reverse=True)
        return [t.params for t in ranked[:self.max_stacking_estimators]]

    def _try_stacking(self, study, X, y):
        """建 StackingRegressor 並用同一套 CV 協定評分，回傳 (model, cv_mean, cv_sem)。"""
        top_params = self._best_params_per_regressor(study)
        if len(top_params) < 2:
            # 只有一種 regressor 表現夠好，湊不出多樣的 base estimator，stacking 沒有意義。
            return None, None, None

        base_estimators = [(p['regressor'], self._build_model(p)) for p in top_params]
        stacking_model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=1.0, random_state=42),
            cv=self._cv(),
        )

        # stacking 內部要對每個 base estimator 做一次 CV 取 out-of-fold 預測，
        # 外層 cross_val_score 再包一層 CV 評分，等於是巢狀 CV，成本頗高——
        # 但這條路只在 n_train > stacking_min_n 時才會走到，是刻意接受的取捨。
        scores = cross_val_score(stacking_model, X, y, cv=self._cv(), scoring=self.scoring, n_jobs=1)
        stacking_model.fit(X, y)
        return stacking_model, float(scores.mean()), self._sem(float(scores.std()), self.cv_n_splits)

    def fit(self, X, y):
        # 使用 Optuna 做 TPE 貝氏優化
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self._objective(trial, X, y), n_trials=self.n_trials)

        print(f"🏆 最佳表現得分: {study.best_value:.4f}")
        print("🔧 最佳模型與參數:", study.best_params)

        self.best_params = study.best_params
        self.best_score = study.best_value

        # 用最佳參數在全部訓練資料上重新訓練一次，這個模型才是要拿去對 testdata 做最終評估的。
        self.best_model_ = self._build_model(study.best_params)
        self.best_model_.fit(X, y)

        n_train = len(X)
        best_mean = study.best_value
        best_sem = self._sem(study.best_trial.user_attrs['cv_std'], self.cv_n_splits)

        self.ensemble_info_ = {
            'n_train': n_train,
            'used_stacking': False,
            'single_best_regressor': study.best_params.get('regressor'),
            'single_best_cv_mean': best_mean,
            'single_best_cv_sem': best_sem,
            'stacking_cv_mean': None,
            'stacking_cv_sem': None,
            'gate_reason': f'n_train={n_train} <= stacking_min_n={self.stacking_min_n}，跳過 stacking（第一道門）',
        }

        # 第一道門：n_train 夠大才值得花算力嘗試 stacking。
        if n_train > self.stacking_min_n:
            stacking_model, stacking_mean, stacking_sem = self._try_stacking(study, X, y)

            if stacking_model is None:
                self.ensemble_info_['gate_reason'] = '表現夠好的 regressor 種類不足 2 種，無法組成 stacking'
            else:
                self.ensemble_info_['stacking_cv_mean'] = stacking_mean
                self.ensemble_info_['stacking_cv_sem'] = stacking_sem

                # 第二道門：stacking 要贏過「雙方 CV 分數變異量」的合理範圍，才算真的比較好。
                margin = self.stacking_significance_k * np.sqrt(stacking_sem ** 2 + best_sem ** 2)
                gap = stacking_mean - best_mean

                if gap > margin:
                    self.best_model_ = stacking_model
                    self.ensemble_info_['used_stacking'] = True
                    self.ensemble_info_['gate_reason'] = (
                        f'stacking 領先單一最佳模型 {gap:.4f} > 顯著性門檻 {margin:.4f}（第二道門通過），採用 stacking')
                else:
                    self.ensemble_info_['gate_reason'] = (
                        f'stacking 領先單一最佳模型 {gap:.4f} 未超過顯著性門檻 {margin:.4f}，差距視為雜訊，維持單一最佳模型')

        return study


def evaluate_regression(y_true, y_pred):
    """算 MSE / RMSE / MAE / Bias / MARD，以及這批 y_true 的最大值、最小值。
    MARD (Mean Absolute Relative Difference) 是血糖預測領域慣用的相對誤差指標，
    Bias 用 (預測 - 實際) 的平均：正值代表模型系統性高估，負值代表系統性低估。"""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    errors = y_pred - y_true

    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))
    # +1e-8 避免 y_true 剛好是 0 時除以 0（血糖值理論上不會是 0，這裡只是保險）
    mard = float(np.mean(np.abs(errors) / (np.abs(y_true) + 1e-8)) * 100)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Bias': bias,
        'MARD': mard,
        'largest_val': float(np.max(y_true)),
        'lowest_val': float(np.min(y_true)),
    }
