import argparse
import sys
import os
data_dir = "/home/eddyliu/alignment/001201/sub-Monkey-N"
neural_decoding_dir = "/home/eddyliu/alignment/neuraldecoding"
# data_dir = "D:/ND/github/LINK_dataset/data/001201/sub-Monkey-N"
# neural_decoding_dir = "D:/ND/github/neuraldecoding"

testname = "all"
N_save_every_day = 10
sys.path.append(neural_decoding_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--config_dir", type=str, default="configs", help="Path to the config directory.")
parser.add_argument("--config", type=str, default="LSTM_NO", help="config file name.")
parser.add_argument("--starting_date_idx", type=int, default=0, help="Starting date index into all_days to begin processing from.")
parser.add_argument("--end_date_idx", type=int, default=None, help="End date index (exclusive). If not provided, processes to the end.")
args = parser.parse_args()

cfg_path = os.path.join(args.config_dir, args.config + "_" + str(args.starting_date_idx))
trialname = os.path.basename(args.config).split('/')[0]
certain_trial_name = args.config + "_" + str(args.starting_date_idx)
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
from neuraldecoding.decoder import NeuralNetworkDecoder
from neuraldecoding.trainer.NeuralNetworkTrainer import IterationNNTrainer
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

days_mat = np.array([
    ["20200924", "20200929", "20201002", "20201016", "20201116", "20210104", "20220207", "20230313"],
    ["20200928", "20201002", "20201007", "20201023", "20201116", "20210106", "20220210", "20230317"],
    ["20201002", "20201007", "20201012", "20201023", "20201116", "20210111", "20220210", "20230320"],
    ["20201012", "20201016", "20201023", "20201105", "20201203", "20210121", "20220224", "20230328"],
    ["20210223", "20210227", "20210305", "20210319", "20210414", "20210604", "20220718", "20230622"],
    ["20210227", "20210305", "20210309", "20210323", "20210420", "20210608", "20220718", "20230622"],
    ["20210305", "20210309", "20210316", "20210330", "20210421", "20210614", "20220718", "20230622"],
    ["20210309", "20210313", "20210319", "20210402", "20210428", "20210617", "20220722", "20230622"],
    ["20210312", "20210316", "20210322", "20210406", "20210503", "20210617", "20220725", "20230622"],
    ["20210313", "20210319", "20210323", "20210407", "20210503", "20210624", "20220725", "20230622"],
    ["20210316", "20210322", "20210323", "20210409", "20210505", "20210624", "20220729", "20230622"],
    ["20210319", "20210323", "20210329", "20210412", "20210505", "20210626", "20220801", "20230622"],
    ["20210323", "20210329", "20210402", "20210415", "20210511", "20210701", "20220805", "20230622"],
    ["20210329", "20210402", "20210407", "20210421", "20210518", "20210707", "20220811", "20230622"],
    ["20210330", "20210405", "20210409", "20210421", "20210519", "20210708", "20220811", "20230622"],
    ["20210402", "20210407", "20210412", "20210428", "20210520", "20210712", "20220817", "20230622"]
])
files = os.listdir(data_dir)
all_days = []
pattern = re.compile(r"sub-Monkey-N_ses-(\d{8})_ecephys\.nwb")
for fname in files:
    match = pattern.match(fname)
    if match:
        all_days.append(match.group(1))
total_days = len(all_days)
all_days.sort()

day_diff = [0,5,10,25,50,100,500,900]

log_mse_total = []
log_r2_total = []
log_r_total = []
fpath_stab_original = conf().preprocessing.preprocessing_trainer.content.stabilization1.params.stabilization_config.params.fpath

original_model_fpath = conf('decoder')["model"]["fpath"]
model_dir = os.path.dirname(original_model_fpath)
model_ext = os.path.splitext(original_model_fpath)[1]
model_name = os.path.splitext(os.path.basename(original_model_fpath))[0]

trial_folder = os.path.join(os.path.dirname(fpath_stab_original), trialname)
if not os.path.exists(trial_folder):
    os.makedirs(trial_folder)

# Slice all_days based on starting/end index args
starting_date_idx = args.starting_date_idx
end_date_idx = args.end_date_idx
days_to_process = all_days[starting_date_idx:end_date_idx] if end_date_idx is not None else all_days[starting_date_idx:]

print(f"\n{'='*60}")
print(f"Starting processing with starting_date_idx = {starting_date_idx} to {end_date_idx if end_date_idx else 'end'}")
print(f"Days to process: {len(days_to_process)} out of {total_days} total days")
print(f"{'='*60}\n")

for i, sel_day in enumerate(days_to_process):
    print(f"==Trial: {certain_trial_name} [starting_idx={starting_date_idx}] Processing day: {sel_day}, day {i + 1} out of {len(days_to_process)} days==")

    sel_day_idx = all_days.index(sel_day)
    days = all_days[sel_day_idx:]
    
    new_stab_fpath = os.path.join(trial_folder, f'{trialname}-{days[0]}.pkl')

    conf.update("preprocessing.preprocessing_trainer.content.stabilization1.params.stabilization_config.params.fpath", new_stab_fpath)
    conf.update("preprocessing.preprocessing_decoder.content.stabilization1.params.stabilization_config.params.fpath", new_stab_fpath)
    data = {'data_path':os.path.join(data_dir, f"sub-Monkey-N_ses-{days[0]}_ecephys.nwb")}
    
    preprocessor_trainer = Preprocessing(conf('preprocessing')['preprocessing_trainer'])
    trainer = IterationNNTrainer(preprocessor_trainer, trainer_config, data)
    # Duplicate stabilization file for each day
    for j, day in enumerate(days):
        source_file = new_stab_fpath
        target_file = os.path.join(trial_folder, f'{trialname}-{days[0]}-{day}.pkl')
        shutil.copy2(source_file, target_file)

    # train
    model, results = trainer.train_model()
    dynamic_model_path = os.path.join(model_dir, f"{model_name}_{days[0]}{model_ext}")
    conf.update("decoder.model.fpath", dynamic_model_path)
    model.save_model(dynamic_model_path)

    log_r = []
    log_r2 = []
    log_mse = []
    total_test_days = len(days)
    pbar = tqdm(enumerate(days[:]), total=total_test_days, desc=f"Testing [start_idx={starting_date_idx}]")
    for j, day in pbar:
        
        # saving stuff for em so principle angle and stuff can be looked at
        new_stab_fpath = os.path.join(trial_folder, f'{trialname}-{days[0]}-{day}.pkl')
        conf.update("preprocessing.preprocessing_trainer.content.stabilization1.params.stabilization_config.params.fpath", new_stab_fpath)
        conf.update("preprocessing.preprocessing_decoder.content.stabilization1.params.stabilization_config.params.fpath", new_stab_fpath)
        
        preprocessor_decoder = Preprocessing(conf('preprocessing')['preprocessing_decoder'])
        decoder = NeuralNetworkDecoder(conf('decoder'))
        decoder.load_model()
        data = {'data_path':os.path.join(data_dir, f"sub-Monkey-N_ses-{day}_ecephys.nwb")}
        
        _, neural_test, _, finger_test = preprocessor_decoder.preprocess_pipeline(data, params = {'is_train': False})

        rr_prediction = decoder.predict(neural_test)
        if isinstance(rr_prediction, torch.Tensor):
            rr_prediction = rr_prediction.detach().cpu().numpy()
        normalizer_path = conf('preprocessing')['preprocessing_trainer']['content']['normalize_standard']['params']['normalizer_params']['save_path']
        with open(normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)
        rr_prediction = normalizer.inverse_transform(rr_prediction)
        r = np.array([pearsonr(finger_test[:, k], rr_prediction[:, k])[0] for k in range(finger_test.shape[1])])
        r2 = r2_score(finger_test, rr_prediction, multioutput='raw_values')
        mse = np.mean((finger_test - rr_prediction)**2, axis=0)

        torch.cuda.empty_cache()

        log_r.append(r)
        log_r2.append(r2)
        log_mse.append(mse)
        pbar.set_postfix({'Trial': trialname, 'Start IDX': starting_date_idx, 'Test Day': day, 'Test Day IDX': j, "Total Test Day": total_test_days,})
        # Create a dataframe row with the current results
        df_row = pd.DataFrame({
            'train_day': [sel_day],
            'test_day': [day],
            'r': [r],
            'r2': [r2],
            'mse': [mse]
        })

        # Initialize results_df if it doesn't exist
        if 'results_df' not in locals():
            csv_path = os.path.join(result_save_path, f"results_{trialname}_{testname}_startOn_{str(starting_date_idx)}.csv")
            if os.path.exists(csv_path):
                results_df = pd.read_csv(csv_path)
            else:
                results_df = pd.DataFrame(columns=['train_day', 'test_day', 'r', 'r2', 'mse'])
        # Append the row to the results dataframe
        results_df = pd.concat([results_df, df_row], ignore_index=True)
        
    if i % N_save_every_day == 0:
        csv_path = os.path.join(result_save_path, f"results_{trialname}_{testname}_startOn_{str(starting_date_idx)}_{i}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"[starting_idx={starting_date_idx}] Saved results for day {sel_day} at day {i}/{len(days_to_process)}")
        
csv_path = os.path.join(result_save_path, f"results_{trialname}_{testname}_startOn_{str(starting_date_idx)}_{i}.csv")
results_df.to_csv(csv_path, index=False)
print(f"[starting_idx={starting_date_idx}] Final Saved results for day {sel_day} at day {i}/{len(days_to_process)}")