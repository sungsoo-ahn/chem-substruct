import argparse
import os
import random

import torch
from torch.optim import Adam

from trainer import Trainer, FilterTrainer
from model import SmilesGenerator, SmilesGeneratorHandler
from util.dataset import load_dataset
from util.char_dict import SmilesCharDictionary

import neptune

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="run", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default="zinc")
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="/home/sungs/workspace/chem-substruct/resource/data/zinc/train.txt",
    )
    parser.add_argument(
        "--vali_dataset_path",
        type=str,
        default="/home/sungs/workspace/chem-substruct/resource/data/zinc/valid.txt",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="/home/sungs/workspace/chem-substruct/resource/data/zinc/test.txt",
    )
    parser.add_argument("--max_smiles_length", type=int, default=80)

    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--lstm_dropout", type=float, default=0.2)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--save_dir", default="./resource/checkpoint/rule-based-training/")

    args = parser.parse_args()

    device = torch.device(0)
    random.seed(0)

    neptune.init(project_qualified_name="sungsahn0215/rule-based-training")
    neptune.create_experiment(name="pretrain", params=vars(args))
    neptune.append_tag(args.dataset)

    char_dict = SmilesCharDictionary(dataset=args.dataset, max_smi_len=args.max_smiles_length)
    train_dataset = load_dataset(char_dict=char_dict, smi_path=args.train_dataset_path)
    vali_dataset = load_dataset(char_dict=char_dict, smi_path=args.vali_dataset_path)
    test_dataset = load_dataset(char_dict=char_dict, smi_path=args.test_dataset_path)

    input_size = max(char_dict.char_idx.values()) + 1
    generator = SmilesGenerator(
        input_size=input_size,
        hidden_size=args.hidden_size,
        output_size=input_size,
        n_layers=args.n_layers,
        lstm_dropout=args.lstm_dropout,
    )
    generator = generator.to(device)
    optimizer = Adam(params=generator.parameters(), lr=args.learning_rate)
    generator_handler = SmilesGeneratorHandler(
        model=generator,
        optimizer=optimizer,
        char_dict=char_dict,
        max_sampling_batch_size=args.batch_size,
    )

    trainer = Trainer(
        char_dict=char_dict,
        train_dataset=train_dataset,
        vali_dataset=vali_dataset,
        test_dataset=test_dataset,
        generator_handler=generator_handler,
        num_steps=args.num_steps,
        log_freq=args.log_freq,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        device=device,
    )

    trainer.train()
