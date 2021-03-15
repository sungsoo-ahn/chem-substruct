import os
from tqdm import tqdm
import torch
import neptune
from util.function import smis_to_actions, canonicalize
from joblib import Parallel, delayed
import pandas as pd

def check_filter(smi, smas):
    mol = Chem.MolFromSmiles(smi)
    patterns = [Chem.MolFromSmarts(sma) for sma in smas]
    pattern_cnts = []
    for pattern in patterns:
        pattern_cnt = len(mol.GetSubstructMatches(pattern))
        pattern_cnts.append(pattern_cnt > 0)

    return pattern_cnts


class Trainer:
    def __init__(
        self,
        char_dict,
        train_dataset,
        vali_dataset,
        test_dataset,
        generator_handler,
        num_steps,
        log_freq,
        batch_size,
        save_dir,
        device,
    ):
        self.generator_handler = generator_handler
        self.num_steps = num_steps
        self.log_freq = log_freq
        self.save_dir = save_dir
        self.device = device
        self.batch_size = batch_size
        self.char_dict = char_dict

        train_action_dataset, _ = smis_to_actions(char_dict=char_dict, smis=train_dataset)
        vali_action_dataset, _ = smis_to_actions(char_dict=char_dict, smis=vali_dataset)
        test_action_dataset, _ = smis_to_actions(char_dict=char_dict, smis=test_dataset)

        train_action_dataset = torch.LongTensor(train_action_dataset)
        vali_action_dataset = torch.LongTensor(vali_action_dataset)
        test_action_dataset = torch.LongTensor(test_action_dataset)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_action_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        self.vali_loader = torch.utils.data.DataLoader(
            dataset=vali_action_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_action_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        os.makedirs(save_dir, exist_ok=True)

    def train(self):
        for step in tqdm(range(self.num_steps)):
            try:
                actions = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                actions = next(train_iter)

            loss = self.generator_handler.train_on_action_batch(actions=actions, device=self.device)

            if (step + 1) % self.log_freq == 0:
                neptune.log_metric("train/loss", loss)
                self.evaluate()
                self.generator_handler.save(self.save_dir)


    def evaluate(self):
        cum_vali_loss = 0.0
        for vali_actions in self.vali_loader:
            with torch.no_grad():
                vali_loss = self.generator_handler.get_loss_on_action_batch(
                    actions=vali_actions, device=self.device
                    )

            cum_vali_loss += vali_loss

        avg_vali_loss = cum_vali_loss / len(self.vali_loader)
        neptune.log_metric("vali/loss", loss)

        cum_test_loss = 0.0
        for test_actions in self.test_loader:
            with torch.no_grad():
                test_loss = self.generator_handler.get_loss_on_action_batch(
                    actions=test_actions, device=self.device
                    )

            cum_test_loss += test_loss

        avg_test_loss = cum_test_loss / len(self.test_loader)
        neptune.log_metric("test/loss", loss)

class FilterTrainer(Trainer):
    def __init__(
        self,
        char_dict,
        train_dataset,
        vali_dataset,
        test_dataset,
        generator_handler,
        num_steps,
        log_freq,
        batch_size,
        save_dir,
        device,
    ):
        super(FilterTrainer, self).__init__(
            char_dict,
            train_dataset,
            vali_dataset,
            test_dataset,
            generator_handler,
            num_steps,
            log_freq,
            batch_size,
            save_dir,
            device,
            )

        self.pool = Parallel(n_jobs=8)
        rules_file_name = "/home/sungs/workspace/chem-substruct/resource/rd_filter/alert_collection.csv"
        rule_df = pd.read_csv(rules_file_name)[["rule_id", "smarts", "max", "rule_set_name", "description"]]
        sma_df = rule_df["smarts"][rule_df["rule_set_name"] == "Inpharmatica"]
        self.smas = sma_df.values.tolist()

    def check_filter(self, smis):
        _check_filter = lambda smi: check_filter(smi, self.smas)
        filter_results = self.pool(delayed(_check_filter)(smi) for smi in smis)
        filter_results = torch.FloatTensor(filter_results).to(self.device)
        return filter_results

    def train(self):
        for step in tqdm(range(self.num_steps)):
            try:
                actions = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                actions = next(train_iter)

            loss0 = self.generator_handler.train_on_action_batch(actions=actions, device=self.device)

            with torch.no_grad():
                self.generator_handler.model.eval()
                sampled_smis, _, _, _ = self.generator_handler.sample(
                    num_samples=self.batch_size, device=self.device
                )

            sampled_smis = self.canonicalize_smis(sampled_smis)

            sampled_actions = smis_to_actions(self.char_dict, sampled_smis)
            weights = self.check_filter(sampled_smis)
            loss1 = self.generator_handler.get_loss_on_action_batch(
                actions=actions, device=self.device, weights=weights
                )

            loss = loss0 + loss1
            self.generator_handler.train_on_loss(loss)

            if (step + 1) % self.log_freq == 0:
                neptune.log_metric("train/loss", loss)
                self.evaluate()

    def canonicalize_smis(self, smis):
        smis = self.pool(
            delayed(lambda smi: canonicalize(smi))(smi) for smi in smis
        )
        smis = list(filter(lambda smi: (smi is not None) and self.char_dict.allowed(smi), smis))
        return smis