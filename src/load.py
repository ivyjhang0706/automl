import os


def load_path_info(basepath, splitting_ratio, uuid, model_type, least_num):
    """依 least_num 門檻決定該 uuid 要不要訓練。

    邏輯取自 Regression_Model_Predictor_meta_v1.py 裡 BuildRegressionModel 對每個
    uuid/model_type 的判斷（原本 num_cv=3 是寫死的下限，這裡改成參數化的 least_num）：
    Train/<model_type> 資料夾樣本數 < least_num，或 Train、Test 任一邊資料夾是空的，
    就回傳 False（main.py 收到後直接 continue、不訓練這個 uuid）。

    回傳 (current_path, current_test_path)：分別指到 Regression_Features/<uuid>/Train、
    /Test 這一層（不含 Normal/High/Low 子資料夾），交給 Regression_ECGDataset 自己
    依 model_type 去讀對應子資料夾。
    """
    current_path = os.path.join(basepath, splitting_ratio, 'Regression_Features', uuid, 'Train')
    if model_type == 'Normal':
        filelist_model_type = os.listdir(os.path.join(current_path, 'Normal'))
    elif model_type == 'High':
        filelist_model_type = os.listdir(os.path.join(current_path, 'High'))
    else:
        filelist_model_type = os.listdir(os.path.join(current_path, 'Low'))

    if len(filelist_model_type) < least_num:  ##如果樣本個數少於 least_num，此類別的回歸模型不訓練
        return False

    current_test_path = os.path.join(basepath, splitting_ratio, 'Regression_Features', uuid, 'Test')
    if model_type == 'Normal':
        filelist_test_model_type = os.listdir(os.path.join(current_test_path, 'Normal'))
    elif model_type == 'High':
        filelist_test_model_type = os.listdir(os.path.join(current_test_path, 'High'))
    else:
        filelist_test_model_type = os.listdir(os.path.join(current_test_path, 'Low'))

    if len(filelist_model_type) == 0 or len(filelist_test_model_type) == 0:
        return False

    return current_path, current_test_path
