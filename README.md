# 特徵預測血糖值 AutoML

`ivyjhang/swmlml:1.0`

用每位使用者的心電圖(ECG)衍生特徵，回歸預測血糖值。每個 uuid(病患/使用者)各自訓練一組模型，
用 Optuna 在多種回歸器之間做超參數搜尋，並視資料量決定要不要進一步做 stacking 集成。

## 目錄結構

```
src/
  main.py     # 進入點：對每個 uuid 跑一輪 AutoML，彙整績效成 csv
  automl.py   # SimpleAutoMLRegressor：Optuna 搜尋 + stacking 兩道門邏輯
  dl_ml.py    # TabNetRegressorWrapper / ResNetMLPRegressor（sklearn 介面包裝）
  data.py     # Regression_ECGDataset：讀取 ECG 特徵檔案
  load.py     # load_path_info：依 least_num 門檻決定該 uuid 要不要訓練
data/
  70_30/Regression_Features/<uuid>/{Train,Test}/{Normal,High,Low}/  # 特徵檔案存放路徑
```

## 執行方式

```bash
python src/main.py
```

跑完後會在專案根目錄產生 `automl_results_summary.csv`，彙整所有 uuid 的訓練/測試結果。

用 container 跑的話請注意：[Dockerfile](Dockerfile) 只把 Python 環境跟套件烤進 image，
**不含這個 repo 的程式碼跟 `data/`**（避免病患資料被烤進 image、且不用每次改程式碼就要重新
build）。程式碼跟資料一律靠 `docker-compose.yml` 的 `/share` bind mount 在執行期提供，
所以主機上的 `/share/automl` 底下要先有這個 repo 的程式碼跟 `data/`，container 才看得到。

## SSH 連進 container

主機的 22 port 已經被主機自己的 sshd 占用，container 的 sshd 改對外開 2222 port
（見 [docker-compose.yml](docker-compose.yml)）。image 裡不會寫死任何密碼，
`root` 帳號預設沒有密碼、無法登入，第一次啟動 container 後要自己手動設一次。

**如果你有主機的 docker 權限**（自己開的機器）：

```bash
docker compose up -d --build              # 啟動 container
docker compose exec automl passwd root    # 互動輸入密碼，不會留在任何檔案裡
```

**如果是用算力平台**（平台自己管 port mapping，不吃 `docker-compose.yml`，
通常只能用平台提供的網頁終端機直接進到 container 裡）：容器內本來就是 root，
直接在 container 裡跑：

```bash
PW=$(openssl rand -base64 12)
echo "root:$PW" | chpasswd
echo "密碼是: $PW"   # 記下來，這裡是唯一會顯示密碼的地方
```

之後就能用 `ssh root@<主機IP或平台給的IP> -p 2222` 登入。**container 只要被重建**
（`docker compose down` 再 `up`、平台重啟 container、或換一台主機）**密碼就會重置**，
要重新跑一次上面其中一種設密碼的指令。

## 本地測試（不用 `/share`）

正式環境的 [docker-compose.yml](docker-compose.yml) 是掛載平台既有的 `/share` 資料夾
（`working_dir: /share/automl`），本地或全新機器上通常沒有這個資料夾。測試時另外疊一份
[docker-compose.local.yml](docker-compose.local.yml)，把這個 repo 本身直接掛進
container 的 `/share/automl`，不用真的建 `/share`：

```bash
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build
```

資料檔案照舊放在這個 repo 底下的 `data/70_30/Regression_Features/<uuid>/...`
（見上面的「目錄結構」）即可。

**正式環境永遠只跑 `docker compose up`**（不帶 `-f docker-compose.local.yml`），
自動就是 `/share` 路徑，不會被本地測試設定影響，也不用擔心忘記切換回來。

## 資料前處理與門檻

- 特徵欄位：`used_feature_dic`（[main.py](src/main.py)）標記 106 個 ECG 衍生特徵中要用哪些（`row[0]==1`）。
- **least_num 門檻**：[load.py](src/load.py) 的 `load_path_info(..., least_num=10)` —— 若某 uuid 在
  `Train/Normal` 資料夾下的樣本數 < 10，或 Train/Test 任一邊資料夾是空的，該 uuid 直接跳過、不訓練
  （[main.py](src/main.py) 收到 `False` 就 `continue`）。
- 特徵中的 inf/NaN（比值特徵分母為 0 造成）由 `InfToNanImputer`（[automl.py](src/automl.py)）處理，
  包在 Pipeline 裡讓補值只用當前 CV fold 的 train 子集平均值，避免 fold 間資訊洩漏。

## 模型搜尋（單一最佳模型）

`SimpleAutoMLRegressor`（[automl.py](src/automl.py)）用 Optuna(TPE) 在 6 種回歸器間搜尋：

| 候選模型 | 說明 |
|---|---|
| RandomForest | `sklearn.ensemble.RandomForestRegressor` |
| XGBoost | `xgboost.XGBRegressor` |
| SVR | `sklearn.svm.SVR` |
| Ridge | `sklearn.linear_model.Ridge` |
| TabNet | `pytorch_tabnet` 包裝成 sklearn 介面（[dl_ml.py](src/dl_ml.py)） |
| ResNetMLP | 自製 Residual MLP，包裝成 sklearn 介面（[dl_ml.py](src/dl_ml.py)） |

- 每個 trial 用 5-fold `KFold` CV（血糖是連續值，不用 `StratifiedKFold`）評分，預設
  `scoring='neg_mean_absolute_percentage_error'`。
- TabNet/ResNetMLP 共用同一張 GPU，CV 內 `n_jobs=1` 避免搶顯存；其餘模型 `n_jobs=-1`。
- 搜尋跑完後，用最佳參數在全部訓練資料上 refit 一次，存進 `self.best_model_`，供 held-out
  testdata 做最終評估。

## Stacking 集成：兩道門設計

單一最佳模型在小樣本時很合理，但 n 夠大時，把 Optuna 搜尋過程中「各種模型表現最好的那組參數」
組成 `StackingRegressor` 可能可以再進一步提升績效。這裡不是無條件都做 stacking，而是用兩道門判斷：

```
n_train > stacking_min_n(600)? ──No──→ 用單一最佳模型（不算 stacking，省算力）
        │Yes
        ▼
   組 StackingRegressor，用同一套 5-fold CV 算出 stacking 的 mean、SEM
        │
        ▼
stacking_mean - best_mean > k * sqrt(stacking_sem² + best_sem²)? ──No──→ 差距視為雜訊，維持單一最佳模型
        │Yes
        ▼
   採用 stacking 當最終模型
```

### 第一道門：`stacking_min_n`（粗篩，省算力）

`n_train` 沒超過 600 就完全不嘗試 stacking。原因：

- Stacking 需要 `cross_val_predict` 幫每個 base estimator 產生 out-of-fold 預測給 meta-learner，
  小樣本下這些 out-of-fold 預測本身就很不穩定，meta-learner 容易過擬合到雜訊。
- TabNet/ResNetMLP 訓練成本高，若真的評估 stacking 的 CV 分數，等於巢狀 CV
  （外層 5-fold 評分 × StackingRegressor 內部 5-fold 產生 meta-feature），成本会再乘一次——
  只有 n 夠大、值得付出這個成本時才嘗試。

### 第二道門：CV 分數的 SEM 顯著性檢定（決定要不要採用）

門檻通過、真的把 stacking 的 CV 分數算出來以後，**不是單純比較平均值**，而是同時考慮兩者 CV
分數的變異量：

- `SEM`（標準誤差）= `std / sqrt(fold數)`，反映「這個平均分數的估計有多準」，而不是原始分數本身
  的離散度。5-fold 在小樣本下每個 fold 驗證集可能只有個位數筆資料，CV 分數波動很大，`SEM` 會如實
  反映這種不可信度。
- 只有當 `stacking_mean - best_mean` 大於 `stacking_significance_k * sqrt(stacking_sem² + best_sem²)`
  （近似 Welch's t-test 的顯著性判斷，預設 `k=1.5`）才採用 stacking；否則差距被視為雜訊，
  維持原本的單一最佳模型。

這樣即使 n_train 超過 600、算出來的 stacking CV 分數也只是「運氣好」，第二道門也會攔下來，
不會讓一個其實沒有真的更好的 stacking 模型被拿去做最終預測。

### 可調參數

`SimpleAutoMLRegressor.__init__` 的相關參數：

| 參數 | 預設值 | 說明 |
|---|---|---|
| `stacking_min_n` | 600 | 第一道門：n_train 超過才嘗試 stacking |
| `stacking_significance_k` | 1.5 | 第二道門：顯著性門檻的寬鬆係數 |
| `max_stacking_estimators` | 4 | stacking 最多取幾種 regressor 當 base estimator |
| `cv_n_splits` | 5 | CV fold 數 |

## 輸出欄位（automl_results_summary.csv）

| 欄位 | 說明 |
|---|---|
| `uuid`, `start_time`, `end_time` | 使用者識別與血糖資料期間 |
| `best_regressor`, `best_params` | Optuna 選出的單一最佳模型與參數 |
| `cv_best_score` | 單一最佳模型的 CV 分數（`neg_mean_absolute_percentage_error`） |
| `n_train`, `n_test` | 訓練/測試筆數 |
| `used_stacking` | 這個 uuid 最終是否採用 stacking（`True`/`False`） |
| `single_best_cv_mean`, `single_best_cv_sem` | 單一最佳模型的 CV 平均分數與 SEM |
| `stacking_cv_mean`, `stacking_cv_sem` | stacking 的 CV 平均分數與 SEM（未嘗試 stacking 時為空） |
| `ensemble_gate_reason` | 兩道門各自判斷的文字說明，方便追蹤某個 uuid 為何用/沒用 stacking |
| `MSE`, `RMSE`, `MAE`, `Bias`, `MARD` | 最終模型在 held-out testdata 上的績效 |
| `largest_val`, `lowest_val` | 該 uuid testdata 的血糖最大值/最小值 |
