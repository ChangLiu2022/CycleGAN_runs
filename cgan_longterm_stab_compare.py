import psutil
import gc
import torch
import os
def log_memory(label=""):
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / 1024 / 1024 / 1024
    gpu_alloc = torch.cuda.memory_allocated() / 1024 / 1024 / 1024 if torch.cuda.is_available() else 0
    gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024 if torch.cuda.is_available() else 0
    print(f"[MEM {label}] CPU: {rss:.2f}GB | GPU alloc: {gpu_alloc:.2f}GB | GPU reserved: {gpu_reserved:.2f}GB")

def main():
    import argparse
    import sys
    import os
    data_dir = "/home/eddyliu/alignment/001201/sub-Monkey-N"
    neural_decoding_dir = "/home/eddyliu/alignment/neuraldecoding"
    # data_dir = "D:/ND/github/LINK_dataset/data/001201/sub-Monkey-N"
    # neural_decoding_dir = "D:/ND/github/neuraldecoding"
    testname = "TENPERCENT-MEMFIX"
    N_save_every_day = 3
    sys.path.append(neural_decoding_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="configs", help="Path to the config directory.")
    parser.add_argument("--config", type=str, default="RR_CGAN_SAVE", help="config file name.")
    parser.add_argument("--day0s", type=str, default=None, help="Comma-separated day0 dates to run, e.g. '20200127,20200930'. If None, runs all days.")
    args = parser.parse_args()
    cfg_path = os.path.join(args.config_dir, args.config)
    trialname = os.path.basename(args.config).split('/')[0]
    result_save_path = os.path.join("results", trialname+ '_' + testname)
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    import re
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    import pickle
    import time
    from omegaconf import OmegaConf
    from sklearn.metrics import r2_score
    from tqdm import tqdm
    from neuraldecoding.decoder import LinearDecoder
    from neuraldecoding.trainer.NeuralNetworkTrainer import NNTrainer
    from neuraldecoding.trainer.LinearTrainer import LinearTrainer
    from neuraldecoding.utils import load_one_nwb
    from neuraldecoding.utils import parse_verify_config
    from neuraldecoding.utils import prep_data_decoder
    from neuraldecoding.preprocessing import Preprocessing
    from neuraldecoding.utils import config
    from neuraldecoding.dataset import Dataset
    from datetime import datetime
    import shutil

    config_dir = os.path.join(os.path.dirname(__file__), cfg_path)
    conf = config(config_dir)
    decoder_config = conf('decoder')
    trainer_config = conf('trainer')
    preprocessing_config = conf('preprocessing')
    preprocessing_trainer_config = preprocessing_config['preprocessing_trainer']
    preprocessing_decoder_config = preprocessing_config['preprocessing_decoder']
    days_mat = np.array([['20200127', '20200130', '20200204', '20200205', '20200206', '20200211', '20200222', '20200224', '20200225', '20200228', '20200302', '20200310', '20200313', '20200317', '20200318', '20200319', '20200320', '20200626', '20200708', '20200714', '20200729', '20200731', '20200808', '20200811', '20200822', '20200829', '20200904', '20200924', '20200928', '20200929', '20201002', '20201007', '20201012', '20201016', '20201023', '20201103', '20201105', '20201109', '20201116', '20201203', '20201204', '20201211', '20201212', '20210104', '20210105', '20210106', '20210108', '20210111', '20210121', '20210122', '20210123', '20210223', '20210225', '20210226', '20210227', '20210305', '20210309', '20210312', '20210313', '20210316', '20210319', '20210322', '20210323', '20210329', '20210330', '20210402', '20210405', '20210406', '20210407', '20210409', '20210412', '20210414', '20210415', '20210420', '20210421', '20210428', '20210503', '20210504', '20210505', '20210511', '20210518', '20210519', '20210520', '20210525', '20210526', '20210601', '20210604', '20210608', '20210614', '20210615', '20210616', '20210617', '20210624', '20210626', '20210628', '20210629', '20210630', '20210701', '20210703', '20210706', '20210707', '20210708', '20210709', '20210712', '20210713', '20210714', '20210715', '20210716', '20210719', '20210720', '20210721', '20210726', '20210727', '20210728', '20210729', '20210730', '20210731', '20210802', '20210803', '20210804', '20210805', '20210810', '20210811', '20210812', '20210816', '20210818', '20210819', '20210820', '20210823', '20210825', '20210827', '20210830', '20210831', '20210903', '20210906', '20210907', '20210908', '20210913', '20210914', '20210915', '20210917', '20210924', '20211001', '20211005', '20211007', '20211008', '20211019', '20211025', '20211026', '20211027', '20211029', '20211101', '20211102', '20211112', '20211116', '20211123', '20211129', '20211215', '20220110', '20220112', '20220114', '20220117', '20220119', '20220120', '20220124', '20220126', '20220128', '20220131', '20220202', '20220203', '20220207', '20220209', '20220210', '20220221', '20220224', '20220228', '20220307', '20220314', '20220316', '20220318', '20220321', '20220323', '20220328', '20220331', '20220401', '20220404', '20220405', '20220406', '20220407', '20220411', '20220413', '20220414', '20220418', '20220420', '20220421', '20220422', '20220425', '20220427', '20220428', '20220509', '20220510', '20220511', '20220523', '20220526', '20220601', '20220602', '20220608', '20220610', '20220613', '20220718', '20220721', '20220722', '20220725', '20220727', '20220729', '20220801', '20220805', '20220808', '20220811', '20220817', '20220818', '20220819', '20220822', '20220825', '20220826', '20220829', '20220831', '20220901', '20220902', '20220906', '20220907', '20220909', '20220912', '20220913', '20220919', '20220923', '20220926', '20220927', '20220930', '20221003', '20221010', '20221012', '20221025', '20221027', '20221101', '20221104', '20221107', '20221109', '20221209', '20230110', '20230113', '20230117', '20230123', '20230124', '20230127', '20230130', '20230131', '20230203', '20230206', '20230207', '20230210', '20230213', '20230214', '20230220', '20230224', '20230306', '20230310', '20230313', '20230314', '20230317', '20230320', '20230328', '20230404', '20230405', '20230407', '20230411', '20230413', '20230414', '20230417', '20230425', '20230426', '20230502', '20230508', '20230517', '20230519', '20230606', '20230608', '20230612', '20230613', '20230616', '20230619', '20230620', '20230621', '20230622']])
    # files = os.listdir(data_dir)
    # all_days = []
    # pattern = re.compile(r"sub-Monkey-N_ses-(\d{8})_ecephys\.nwb")
    # for fname in files:
    #     match = pattern.match(fname)
    #     if match:
    #         all_days.append(match.group(1))
    all_days = days_mat.flatten()

    if args.day0s is not None:
        selected_day0s = args.day0s.split(',')
    else:
        selected_day0s = all_days.tolist()

    total_days = len(selected_day0s)
    # all_days.sort()

    # days_mat = np.array([
    #     ["20200924", "20200929", "20201002", "20201016", "20201116", "20210104", "20220207", "20230313"],
    #     ["20201012", "20201016", "20201023", "20201105", "20201203", "20210121", "20220224", "20230328"],
    #     ["20210312", "20210316", "20210322", "20210406", "20210503", "20210617", "20220725", "20230622"],
    #     ["20210330", "20210405", "20210409", "20210421", "20210519", "20210708", "20220811", "20230622"]
    # ])
    # days_mat = np.array([
    #     ["20200924", "20200929", "20201002", "20201016", "20201116", "20210104", "20220207", "20230313"]
    # ])
    day_diff = [0,5,10,25,50,100,500,900]

    log_mse_total = []
    log_r2_total = []
    log_r_total = []

    results_df = None

    for i, sel_day in enumerate(selected_day0s):
        day0_idx = np.where(all_days == sel_day)[0]
        if len(day0_idx) == 0:
            print(f"WARNING: day {sel_day} not found in days_mat, skipping")
            continue
        day0_idx = day0_idx[0]
        test_days = all_days[day0_idx:]

        print(f"==Trial: {trialname} Processing day0: {sel_day}, day {i + 1} out of {total_days}, testing on {len(test_days)} future days==")

        conf.update("decoder.model.fpath", conf().decoder.model.fpath.format(date=sel_day))
        data = {'data_path': os.path.join(data_dir, f"sub-Monkey-N_ses-{sel_day}_ecephys.nwb")}
        preprocessor_trainer = Preprocessing(conf('preprocessing')['preprocessing_trainer'])
        trainer = LinearTrainer(preprocessor_trainer, trainer_config, data)
        
        # train
        model, results = trainer.train_model()
        model_path = conf('decoder')["model"]["fpath"]
        model.save_model(model_path)

        # preprocessor_decoder = Preprocessing(preprocessing_decoder_config)
        # decoder = LinearDecoder(decoder_config)
        # decoder.load_model()
        log_r = []
        log_r2 = []
        log_mse = []
        total_test_days = len(test_days)
        pbar = tqdm(enumerate(test_days), total=total_test_days)
        for j, day in pbar:
            log_memory(f"start {sel_day}_{day}")

            preprocessor_decoder = Preprocessing(conf('preprocessing')['preprocessing_decoder'])
            decoder = LinearDecoder(decoder_config)
            decoder.load_model()
            data = {'data_path':os.path.join(data_dir, f"sub-Monkey-N_ses-{day}_ecephys.nwb")}

            _, neural_test, _, finger_test = preprocessor_decoder.preprocess_pipeline(data, params = {'is_train': False, 'decoder': decoder})
            
            log_memory(f"after pipeline {sel_day}_{day}")
            
            rr_prediction = decoder.predict(neural_test)

            normalizer_path = conf('preprocessing')['preprocessing_trainer']['content']['normalize_standard']['params']['normalizer_params']['save_path']
            with open(normalizer_path, 'rb') as f:
                normalizer = pickle.load(f)
            rr_prediction = normalizer.inverse_transform(rr_prediction)
            r = np.array([pearsonr(finger_test[:, k], rr_prediction[:, k])[0] for k in range(finger_test.shape[1])])
            r2 = r2_score(finger_test, rr_prediction, multioutput='raw_values')
            mse = np.mean((finger_test - rr_prediction)**2, axis=0)

            torch.cuda.empty_cache()
            # print(f"day {day_diff[i]}")
            # print(f"    R for day {day_diff[i]}: {r}")
            # print(f"    R2 for day {day_diff[i]}: {r2}")
            # print(f"    MSE for day {day_diff[i]}: {mse}")

            log_r.append(r)
            log_r2.append(r2)
            log_mse.append(mse)
            # print(f"Trial: {trialname}")
            # print(f"    =Training day: {sel_day}, day {i + 1} out of {total_days} days=")
            # print(f"    =Testing day: {day}, day {j + 1} out of {total_test_days} days=")
            # print(f"        r: {r},\n        r2: {r2},\n        mse: {mse}")
            pbar.set_postfix({'Trial': trialname, 'Test Day': day, 'Test Day IDX': j, "Total Test Day": total_test_days,})
            # Create a dataframe row with the current results
            df_row = pd.DataFrame({
                'train_day': [sel_day],
                'test_day': [day],
                'r': [r],
                'r2': [r2],
                'mse': [mse]
            })

            # Initialize results_df if it doesn't exist
            if results_df is None:
                csv_path = os.path.join(result_save_path, f"results_{trialname}_{testname}.csv")
                if os.path.exists(csv_path):
                    results_df = pd.read_csv(csv_path)
                else:
                    results_df = pd.DataFrame(columns=['train_day', 'test_day', 'r', 'r2', 'mse'])
            # Append the row to the results dataframe
            results_df = pd.concat([results_df, df_row], ignore_index=True)

            # del preprocessor_decoder, decoder, data, neural_test, finger_test, rr_prediction
            # torch.cuda.empty_cache()
            # gc.collect()
            log_memory(f"after cleanup {sel_day}_{day}")
            
        if i % N_save_every_day == 0:
            csv_path = os.path.join(result_save_path, f"results_{trialname}_{testname}_{i}.csv")
            results_df.to_csv(csv_path, index=False)
            print(f"Saved results for day {sel_day} at day {i}/{total_days}")
            
    csv_path = os.path.join(result_save_path, f"results_{trialname}_{testname}_{i}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Final Saved results for day {sel_day} at day {i}/{total_days}")

if __name__ == '__main__':
    main()