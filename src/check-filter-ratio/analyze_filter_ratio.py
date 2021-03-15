import numpy as np
import pandas as pd
import pickle

from rdkit import Chem
from joblib import Parallel, delayed
from tqdm import tqdm

def check_filter(smi, smas):
    mol = Chem.MolFromSmiles(smi)
    patterns = [Chem.MolFromSmarts(sma) for sma in smas]
    pattern_cnts = []
    for pattern in patterns:
        pattern_cnt = len(mol.GetSubstructMatches(pattern))
        pattern_cnts.append(pattern_cnt)

    return pattern_cnts


if __name__ == "__main__":
    rules_file_name = "./resource/data/rd_filter/alert_collection.csv"
    rule_df = pd.read_csv(rules_file_name)[["rule_id", "smarts", "max", "rule_set_name", "description"]]
    sma_df = rule_df["smarts"]
    smas = sma_df.values.tolist()
    
    smis_path = "./resource/data/chembl/all.txt"
    with open(smis_path, "r") as f:
        smis = f.read().splitlines()  
        
    _check_filter = lambda smi: check_filter(smi, smas) 
    pool = Parallel(n_jobs=16)
    filter_results = pool(delayed(_check_filter)(smi) for smi in smis)
    filter_results = np.array(filter_results)
    
    with open("./resource/result/chembl_all.pkl", "wb") as f:
        pickle.dump(filter_results, f)