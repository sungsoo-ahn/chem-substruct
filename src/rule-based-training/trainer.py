import os
from tqdm import tqdm
import torch
import neptune
from util.function import smis_to_actions


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
                neptune.log_metric("loss/train", loss)
                
                cum_vali_loss = 0.0
                for vali_actions in self.vali_loader:
                    vali_loss = self.generator_handler.loss_on_action_batch(actions=vali_actions, device=self.device)
                    cum_vali_loss += vali_loss
                    
                avg_vali_loss = cum_vali_loss / len(self.vali_loader)
                neptune.log_metric("loss/vali", loss)
                
