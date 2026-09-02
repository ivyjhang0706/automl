import json
import load
from data import Regression_ECGDataset
from pathlib import Path
import numpy as np
import pandas as pd
from automl import SimpleAutoMLRegressor, evaluate_regression

# 測試用例
if __name__ == '__main__':

    user_information = [
                        ['2197','20250212','20250225'],       ##T1003(績效良，但可再訓練)
                        ['2200','20250212','20250226'],       ##T1005(績效優)
                        ['2133','20250212','20250225'],       ##T1006(績效可，70%左右)
                        ['2131','20250212','20250225'],       ##T1007(績效良，但可再訓練)
                        ['2208','20250217','20250302'],       ##T1011(績效良，但可再訓練)
                        ['2205','20250217','20250302'], ##5   ##T1014(績效良，但可再訓練，注意血糖值有到500的情況)
                        ['2202','20250217','20250302'],       ##T1015(績效良，但可再訓練)
                        ['2215','20250219','20250304'],       ##T1020(績效優)
                        ['2223','20250226','20250311'],       ##T1026(績效良)
                        ['2230','20250226','20250311'],       ##T1028(績效超級優)
                        ['2235','20250312','20250326'], ##10  ##T1032(績效普通)
                        ['2246','20250312','20250320'],       ##T1034(績效良，可在訓練)
                        ['2249','20250313','20250326'],       ##T1035(績效普通，需再訓練)
                        ['2253','20250320','20250402'],       ##T1037(績效普通，要再訓練)
                        ['2261','20250401','20250414'],       ##T1038(績效尚可，還可再訓練)
                        ['2262','20250408','20250421'], ##15  ##T1039(績效普通，train and test各有2筆血糖資料,已使用special move測試)       ##待測試
                        ['2257','20250408','20250421'],       ##T1040(績效普通，可再訓練)


                        ['2199','20250212','20250225'],         ##T1001(績效尚可,使用special move後有進步)
                        ['2198','20250212','20250225'],         ##T1004(績效極度不平衡)                                  ##待測試
                        ['2196','20250212','20250226'], ##19    ##T1002(高血糖只有2筆,未能訓練)
                        ['2206','20250217','20250228'],         ##T1008(test只有兩筆，績效尚可有點不平衡可再訓練或觀察)
                        ['2201','20250217','20250302'],         ##T1009  ##可能只做到2/28(無高、低血糖資料)
                        ['2204','20250217','20250228'],         ##T1010(test high 只有一筆，績效不好可再訓練或觀察，spcial move有進步)
                        ['2203','20250217','20250302'],         ##T1012(train中的高血糖資料只有兩筆，績效普通，可再訓練或觀察)              ##可再次測試
                        ['2207','20250217','20250302'],         ##T1013(special move後績效依然不好)
                        ['2210','20250217','20250302'], ##25    ##T1016(test的高血糖只有1筆，績效不好，已經測試2次)
                        ['2209','20250219','20250304'],         ##T1017(test的高血糖只有2天4筆，test and train個只有2筆, 已經執行過4次訓練)
                        ['2218','20250219','20250304'],         ##T1018(高血糖資料很多，但績效不好過度失衡，還沒使用specail move測試過)
                        ['2219','20250219','20250304'],         ##T1021(只有中、低血糖，無高血糖資料，已訓練回歸)                                       ##要能練low and normal
                        ['2217','20250219','20250304'],         ##T1022(只有中、低血糖，無高血糖資料，已訓練回歸)                                       ##要能練low and normal
                        ['2216','20250219','20250304'], ##30    ##T1023(績效依然不好，special move測試多次)
                        ['2214','20250219','20250304'],         ##T1024(高血糖資料train 只有2筆，test沒有資料，已使用specail move，low血糖很多)  ##要能練low and normal
                        ['2226','20250226','20250312'],         ##T1025(高血糖只有2筆且在同一天,未能處理)
                        ['2224','20250226','20250311'],         ##T1027(高血糖資料很少只有3筆，績效不好)                                    ##待測試
                        ['2227','20250226','20250312'],         ##T1029(高血糖資料很多，但績效差，不平衡，使用specail move測試)
                        ['2225','20250226','20250312'], ##35    ##T1030(高血糖只有1筆、低血糖資料只有4筆)                                  ##待測試
                        ['2236','20250310','20250323'],         ##T1031(高血糖資料只有1筆)                                                 ##待測試
                        ['2248','20250312','20250325'],         ##T1033(績效不好，需再訓練，更新後績效變差，要特別注意)                        ##待測試
                        ['2251','20250319','20250401'],         ##T1036(績效極不平衡，需再訓練)                                              ##待測試
                        # ['2221','20250219','20250220'],         ##提早退出


                        ['2279','20250426','20250509'], ##40    ##T1041
                        ['2276','20250426','20250509'],         ##T1042
                        ['2278','20250426','20250509'],         ##T1043
                        ['2281','20250502','20250515'],         ##T1044
                        ['2286','20250509','20250522'],         ##T1045
                        ['2285','20250509','20250522'], ##45    ##T1046
                        ['2294','20250520','20250531'],         ##T1047
                        ['2293','20250520','20250602'],         ##T1048
                        ['2295','20250522','20250604'],         ##T1049
                        ['2291','20250527','20250610'],         ##T1050
                        ['2298','20250604','20250617'],  ##50   ##T1051(19歲，小於20歲)
                        ['2300','20250604','20250617'],         ##T1052
                        ['2299','20250604','20250617'],         ##T1053  ##訓練到一半出問題
                        ['2304','20250606','20250619'],         ##T1054
                        ['2305','20250606','20250619'],         ##T1055
                        ['2306','20250606','20250619'],  ##55   ##T1056
                        ['2130','20250623','20250706'],         ##T1057  ##test沒有高血糖
                        ['2132','20250623','20250706'],         ##T1058
                        ['2322','20250623','20250706'],  ##58   ##T1059
                        ['2329','20250701','20250713'],         ##T1060
                        ['2352','20250731','20250813'],         ##T1061
                        ['2378','20251022','20251104'],         ##T1062
                        ['2397','20251129','20251213']          ##T1063
                        ]

    used_feature_dic=[(0,'uuid'),(0,'type'),(1,'rr_interval'),(0,'hr'),(0,'a_score'),(0,'p_score'),(0,'r_score'),(0,'p_stability'),(0,'q_stability'),
                    (0,'s_stability'),(0,'t_stability'),(0,'p_value'),(0,'q_value'),(0,'r_value'),(0,'s_value'),(0,'t_value'),(0,'pr_duration'),
                    (0,'pr_amplitude'),(0,'pr_distances'),(0,'pr_directions'),(0,'pr_slope'),(0,'pr_corrections3'),(0,'qr_duration'),(0,'qr_amplitude'),
                    (0,'qr_distances'),(0,'qr_directions'),(0,'qr_slope'),(0,'qr_corrections3'),(0,'rs_duration'),(0,'rs_amplitude'),(0,'rs_distances'),
                    (0,'rs_directions'),(0,'rs_slope'),(0,'rs_corrections3'),(1,'rt_duration'),(0,'rt_amplitude'),(0,'rt_distances'),(0,'rt_directions'),
                    (0,'rt_slope'),(0,'rt_corrections3'),(0,'pq_duration'),(0,'pq_amplitude'),(0,'pq_distances'),(0,'pq_directions'),(0,'pq_slope'),
                    (0,'pq_corrections3'),(0,'ps_duration'),(0,'ps_amplitude'),(0,'ps_distances'),(0,'ps_directions'),(0,'ps_slope'),(0,'ps_corrections3'),
                    (1,'pt_duration'),(0,'pt_amplitude'),(0,'pt_distances'),(0,'pt_directions'),(0,'pt_slope'),(0,'pt_corrections3'),(0,'qs_duration'),
                    (0,'qs_amplitude'),(0,'qs_distances'),(0,'qs_directions'),(0,'qs_slope'),(0,'qs_corrections3'),(1,'qt_duration'),(0,'qt_amplitude'),
                    (1,'qt_distances'),(0,'qt_directions'),(0,'qt_slope'),(0,'qt_corrections3'),(0,'st_duration'),(0,'st_amplitude'),(0,'st_distances'),
                    (0,'st_directions'),(0,'st_slope'),(1,'st_corrections3'),(0,'p_left_slope'),(0,'p_right_slope'),(0,'p_left_sharp'),(0,'p_right_sharp'),
                    (0,'p_tilt'),(0,'r_left_slope'),(0,'r_right_slope'),(0,'r_left_sharp'),(0,'r_right_sharp'),(0,'r_tilt'),(1,'t_left_slope'),(1,'t_right_slope'),
                    (1,'t_left_sharp'),(1,'t_right_sharp'),(0,'t_tilt'), (1,'qrs_area'),(1,'st_area'),(1,'twave_cog') ,(0,'Dataset'),(0,'BG_Level'),(1,'ratio of dif_qs_amp/dif_qr_amp'),(1,'ratio of dif_tr_amp/dif_st_amp'),
                    (1,'ratio of tr_amp'),(1,'ratio of st_amp')]

    mother_path=Path(__file__).resolve().parent.parent   ###血糖數值對應之ECG資料，分析得到特徵檔案會存放在此路徑
    base_path=mother_path/"data"
    base_path=str(base_path)
    user_num = len(user_information)

    # 每個 uuid 跑完就馬上寫一行進 csv，避免中途某個 uuid 出錯時，前面已經跑完的結果全部遺失。
    results_csv_path = mother_path / 'output' / 'automl_results_summary.csv'
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if results_csv_path.exists():
        results_csv_path.unlink()  # 每次重新執行都是全新的一份，不要接續上一次跑到一半的結果

    for i in range(0,user_num):
        user_info=user_information[i]
        ##-------step 1. 基本設定--------
        uuid=user_info[0]
        start_time=user_info[1]  ##第一筆血糖資料記錄日期
        end_time=user_info[2]    ##最後一筆血糖資料紀錄日期
        print('index:',i,' uuid:',uuid)


        used_feature_array = np.array([row[0] for row in used_feature_dic]) # Regression_ECGDataset 會使用 row[0]==1 當作特徵欄位
        path_info = load.load_path_info(base_path, '70_30', uuid, 'Normal', 10)
        if not path_info:
            print(f'  uuid {uuid} 樣本數不足或缺資料，略過')
            continue
        current_path, current_test_path = path_info
        model_type = 'Normal'
        method = 'raw'

        traindata = Regression_ECGDataset(current_path, used_feature_array, type=model_type, method=method)
        testdata = Regression_ECGDataset(current_test_path, used_feature_array, type=model_type, method=method)

        automl = SimpleAutoMLRegressor(n_trials=20, scoring='neg_mean_absolute_percentage_error')
        # Regression_ECGDataset 在 raw method 下會多出一個 channel 維度 (n, 1, n_features)，
        # 我們的模型都吃 2D (n_samples, n_features)，攤平掉多的那一維。
        train_X = traindata.Signals.numpy().reshape(len(traindata), -1)
        train_y = traindata.Labels.numpy()
        study = automl.fit(train_X, train_y)

        ##-------step 2. 用最佳模型在 held-out 的 testdata 上算最終績效--------
        test_X = testdata.Signals.numpy().reshape(len(testdata), -1)
        test_y = testdata.Labels.numpy()
        test_pred = automl.best_model_.predict(test_X)
        metrics = evaluate_regression(test_y, test_pred)

        print(f"  test set 績效: MAE={metrics['MAE']:.2f} RMSE={metrics['RMSE']:.2f} "
              f"MARD={metrics['MARD']:.1f}% Bias={metrics['Bias']:.2f}")
        print(f"  {automl.ensemble_info_['gate_reason']}")

        ensemble_info = automl.ensemble_info_

        row = {
            'uuid': uuid,
            'start_time': start_time,
            'end_time': end_time,
            'best_regressor': study.best_params.get('regressor'),
            'best_params': json.dumps(study.best_params, ensure_ascii=False),
            'cv_best_score': study.best_value,
            'n_train': len(traindata),
            'n_test': len(testdata),
            'used_stacking': ensemble_info['used_stacking'],
            'single_best_cv_mean': ensemble_info['single_best_cv_mean'],
            'single_best_cv_sem': ensemble_info['single_best_cv_sem'],
            'stacking_cv_mean': ensemble_info['stacking_cv_mean'],
            'stacking_cv_sem': ensemble_info['stacking_cv_sem'],
            'ensemble_gate_reason': ensemble_info['gate_reason'],
            **metrics,
        }

        ##-------step 3. 這個 uuid 一跑完就馬上寫入 csv，不用等全部 uuid 跑完--------
        pd.DataFrame([row]).to_csv(
            results_csv_path, mode='a', index=False,
            header=not results_csv_path.exists(), encoding='utf-8-sig',
        )
        print(f'  已寫入 {uuid} 的結果到: {results_csv_path}')

    print(f'\n全部 uuid 執行完畢，結果已彙整到: {results_csv_path}')
