#!/usr/bin/env python3

"""
Minimal bayes inverse example for SmolLM2-135M-Instruct.

Filepath: ./examples/bayes_inverse_example.py
Project: CPEN455-Project-2025W1
Description: integrates three different ways to perform bayes inverse classification with LLMs for spam detection.

Usage:
    uv run -m examples.bayes_inverse
"""

import os
import pdb
import wandb
from dotenv import load_dotenv
from einops import rearrange
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from autograder.dataset import CPEN455_2025_W1_Dataset, ENRON_LABEL_INDEX_MAP, prepare_subset
from dataset.dataset_wrapper import CPEN455DatasetWrapper, split_train_val
from model import LlamaModel
from model.lora_llama import LoraLlamaModel, LoRaClassifer
from utils.weight_utils import load_model_weights
from model.config import Config
from model.tokenizer import Tokenizer
from utils.download import _resolve_snapshot_path
from utils.device import set_device
from utils.prompt_template import get_prompt
from utils.logger import avg_logger, avg_acc_logger

    
def get_seq_log_prob(prompts, tokenizer, model, device):
    encoded_batch = tokenizer.encode(
        prompts, return_tensors="pt", return_attention_mask=True
    )
    input_ids = encoded_batch["input_ids"].to(device)
    attention_mask = encoded_batch["attention_mask"].to(device)

    log_prob, _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    
    shifted_log_prob = log_prob[:, :-1, :]
    shifted_input_ids = input_ids[:, 1:]
    shifted_attention_mask = attention_mask[:, 1:]

    gathered_log_prob = shifted_log_prob.gather(-1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
    gathered_log_prob = gathered_log_prob * shifted_attention_mask
    
    return gathered_log_prob.mean(dim=-1)


METHOD_SET = ["zero_shot", "naive_prompting", "full_finetune", "lora"]

def is_required_training(method: str) -> bool:
    assert method in METHOD_SET, f"Method {method} not recognized. Choose from {METHOD_SET}."
    return method in METHOD_SET[2:]



def discriminative_test(args, model, tokenizer, batch, optimizer=None, is_training=True,
                       log_p_spam=torch.tensor(0.5).log(), log_p_ham=torch.tensor(0.5).log()):
    if is_training:
        model.train()
    else:
        model.eval()

    _, subjects, messages, label_indexs = batch
    
    no_label = -1 in label_indexs

    with torch.no_grad():
        prompts = [get_prompt(subject=subj, message=msg, max_seq_length=args.max_seq_len, label="") for subj, msg in zip(subjects, messages)]

    encoded_batch = tokenizer.encode(
        prompts, return_tensors="pt", return_attention_mask=True
    )
    input_ids = encoded_batch["input_ids"].to(device)
    attention_mask = encoded_batch["attention_mask"].to(device)

    log_prob, _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    # Now, get the posterior probabilities for predicting spam and ham
    softmax_logits = log_prob
    # softmax_logits = torch.stack((spam_seq_log_prob, ham_seq_log_prob), dim=1)

    ce_loss = torch.nn.CrossEntropyLoss()
    
    if not no_label:
        loss_val = ce_loss(softmax_logits, label_indexs.to(device))
    else:
        loss_val = None
    
    if is_training and not no_label:
        assert optimizer is not None, "Optimizer must be provided during training."
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    
    probs = F.softmax(softmax_logits, dim=-1)
    # Get prediction labels
    labels_pred = torch.argmax(probs, dim=-1)
    
    if -1 in label_indexs:
        is_correct = None
    else:
        is_correct = labels_pred.detach().cpu() == label_indexs

    # is_correct, (probs, labels_pred) = bayes_inverse_llm_classifier(args, model, batch, tokenizer, device=device)

    return loss_val, is_correct, (probs.detach().cpu(), labels_pred.detach().cpu())


def cross_entropy_test(args, model, tokenizer, batch, optimizer=None, is_training=True,
                       log_p_spam=torch.tensor(0.5).log(), log_p_ham=torch.tensor(0.5).log()):
    if is_training:
        model.train()
    else:
        model.eval()

    _, subjects, messages, label_indexs = batch
    
    no_label = -1 in label_indexs

    with torch.no_grad():
        spam_prompts = [get_prompt(subject=subj, message=msg, label=ENRON_LABEL_INDEX_MAP.inv[0], max_seq_length=args.max_seq_len) for subj, msg in zip(subjects, messages)]
        ham_prompts = [get_prompt(subject=subj, message=msg, label=ENRON_LABEL_INDEX_MAP.inv[1], max_seq_length=args.max_seq_len) for subj, msg in zip(subjects, messages)]

    if is_training:
        # Count labels in this batch
        batch_spam = len([x for x in label_indexs if x == 1])
        batch_ham  = len([x for x in label_indexs if x == 0])

        if isinstance(model, LoraLlamaModel):
            # Update cumulative counts
            model.count_spam += batch_spam
            model.count_ham  += batch_ham

            total = model.count_spam + model.count_ham

            # Compute priors using cumulative counts (correct)
            log_p_true_spam = torch.log(torch.tensor((model.count_spam + 1) / (total + 2)))
            log_p_true_ham  = torch.log(torch.tensor((model.count_ham  + 1) / (total + 2)))
        else:
            log_p_true_spam = torch.tensor(0.5).log()
            log_p_true_ham = torch.tensor(0.5).log()

        # VALIDATION ONLY — no updates!
    else:
        if isinstance(model, LoraLlamaModel):
            total = model.count_spam + model.count_ham
            log_p_true_spam = torch.log(torch.tensor((model.count_spam + 1) / (total + 2)))
            log_p_true_ham  = torch.log(torch.tensor((model.count_ham  + 1) / (total + 2)))
        else:
            log_p_true_spam = torch.tensor(0.5).log()
            log_p_true_ham  = torch.tensor(0.5).log()
        
    print(model.count_ham, model.count_spam, model.count_ham/(model.count_ham+model.count_spam))


    # shape: 1xd for both spam and ham
    spam_seq_log_prob = get_seq_log_prob(spam_prompts, tokenizer, model, device=device)
    ham_seq_log_prob = get_seq_log_prob(ham_prompts, tokenizer, model, device=device)

    spam_joint_prob = log_p_true_spam.to(device) + spam_seq_log_prob
    ham_joint_prob = log_p_true_ham.to(device) + ham_seq_log_prob

    # Now, get the posterior probabilities for predicting spam and ham
    softmax_logits = torch.stack((spam_joint_prob, ham_joint_prob), dim=1)
    # softmax_logits = torch.stack((spam_seq_log_prob, ham_seq_log_prob), dim=1)

    # pdb.set_trace()
    ce_loss = torch.nn.CrossEntropyLoss()
    
    if not no_label:
        loss_val = ce_loss(softmax_logits, label_indexs.to(device))
    else:
        loss_val = None
    
    if is_training and not no_label:
        assert optimizer is not None, "Optimizer must be provided during training."
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    
    probs = F.softmax(softmax_logits, dim=-1)
    # Get prediction labels
    labels_pred = torch.argmax(probs, dim=-1)
    
    if -1 in label_indexs:
        is_correct = None
    else:
        is_correct = labels_pred.detach().cpu() == label_indexs

    

    # is_correct, (probs, labels_pred) = bayes_inverse_llm_classifier(args, model, batch, tokenizer, device=device)

    return loss_val, is_correct, (probs.detach().cpu(), labels_pred.detach().cpu())


def bayes_inverse_llm_classifier(args, model, batch, tokenizer, device):
    

    _, subjects, messages, labels = batch

    prompts_ham = [get_prompt(subject=subj, message=msg, label=ENRON_LABEL_INDEX_MAP.inv[0], max_seq_length=args.max_seq_len, user_prompt=args.user_prompt) for subj, msg in zip(subjects, messages)]
    prompts_spam = [get_prompt(subject=subj, message=msg, label=ENRON_LABEL_INDEX_MAP.inv[1], max_seq_length=args.max_seq_len, user_prompt=args.user_prompt) for subj, msg in zip(subjects, messages)]

    # The first half are ham, the second half are spam
    prompts = prompts_ham + prompts_spam
    with torch.no_grad():
        seq_log_prob = get_seq_log_prob(prompts, tokenizer, model, device)

        '''
        Rearrange to (batch_size, 2), in this way, the second dimension 0 is ham, 1 is spam.
        '''
        seq_log_prob = rearrange(seq_log_prob, '(c b) -> b c', c=2)
        
        '''
        Apply softmax over ham/spam dimension to get probabilities.
        The shape of probs will be (2, batch_size), where probs[0, :] is ham probability and probs[1, :] is spam probability.
        probs[:, i] gives the category distribution used to classify spam and ham for the i-th email in the batch.
        '''
        probs = F.softmax(seq_log_prob, dim=-1)

        labels_pred = torch.argmax(probs, dim=-1)
        
        if  -1 in labels:
            is_correct = None
        else:
            is_correct = labels_pred.cpu() == labels

        return is_correct, (probs.detach().cpu(), labels_pred.detach().cpu())

def train_or_test(args, model, tokenizer, batch, optimizer=None, is_training=True):
    if is_training:
        model.train()
    else:
        model.eval()

    _, subjects, messages, label_indexs = batch
    
    if -1 in label_indexs:
        bpd = None
    else:
        labels_text = [ENRON_LABEL_INDEX_MAP.inv[int(label_index)] for label_index in label_indexs]

        prompts = [get_prompt(subject=subj, message=msg, label=label, max_seq_length=args.max_seq_len) for subj, msg, label in zip(subjects, messages, labels_text)]

        seq_log_prob = get_seq_log_prob(prompts, tokenizer, model, device=device)
    
        # is_correct, (probs, labels_pred) = bayes_inverse_llm_classifier(args, model, batch, tokenizer, device=device)
            
        num_characters = torch.tensor([len(prompt) for prompt in prompts], device=device).sum()
        bpd = -seq_log_prob.sum()/num_characters

        if is_training:
            assert optimizer is not None, "Optimizer must be provided during training."
            optimizer.zero_grad()
            bpd.backward()
            optimizer.step()

    is_correct, (probs, labels_pred) = bayes_inverse_llm_classifier(args, model, batch, tokenizer, device=device)

    return bpd, is_correct, (probs, labels_pred)

def save_probs(args, model, tokenizer, dataloader, device, name = "test", eval=True):
    save_path = os.path.join(os.getcwd(), f"{args.prob_output_folder}/{name}_dataset_probs.csv")
    
    if os.path.exists(save_path):
        os.remove(save_path)
        
    total_count = 0
    correct_count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="saving probabilities"):
            
            loss, is_correct, (probs, pred_label) = discriminative_test(args, model, tokenizer, batch, is_training=False)
            data_index, _, _, true_label = batch
            indices = torch.as_tensor(data_index).view(-1).tolist()
            
            if(eval):
                total_count += len(is_correct)
                correct_count += is_correct.sum()
                
                    
            rows = zip(indices, probs[:, 0].tolist(), probs[:, 1].tolist())
            file_exists = os.path.exists(save_path)
            with open(save_path, "a", newline="") as handle:
                if not file_exists:
                    handle.write("data_index,prob_ham,prob_spam\n")
                handle.writelines(f"{idx},{ham},{spam}\n" for idx, ham, spam in rows)
        if (eval):
                print(f"[{name}] Accuracy: {correct_count/total_count}")

if __name__ == "__main__":
    # random seed for reproducibility
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="zero-shot", choices=METHOD_SET)
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--dataset_path", type=str, default="autograder/cpen455_released_datasets/train_val_subset.csv")
    parser.add_argument("--test_dataset_path", type=str, default="autograder/cpen455_released_datasets/test_subset.csv")
    parser.add_argument("--prob_output_folder", type=str, default="bayes_inverse_probs")
    parser.add_argument("--user_prompt", type=str, default="")
    
    # Training hyperparameters
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lora_dim", type=int, default= 16)
    parser.add_argument("--lora_sigma", type=float, default= 1.0)
    args = parser.parse_args()

    load_dotenv()
    
    checkpoint = os.getenv("MODEL_CHECKPOINT")
    model_cache_dir = os.getenv("MODEL_CACHE_DIR")
    
    run = None
    if not is_required_training(args.method):
        run = wandb.init(
            project=os.getenv("PROJECT_NAME"), 
            name=f"bayes-inverse-{args.method}_msl{args.max_seq_len}",
        )
    else:
        run = wandb.init(
            project=os.getenv("PROJECT_NAME"), 
            name=f"bayes-inverse-{args.method}_msl{args.max_seq_len}_ni{args.num_iterations}_bs{args.batch_size}",
        )
        
    wandb.config.update(args)

    # Set device to GPU if available, to MPS if on Mac with M-series chip, else CPU
    device = set_device()

    # Load tokenizer and config
    tokenizer = Tokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir)
    
    base_path = _resolve_snapshot_path(checkpoint, cache_dir=model_cache_dir)
    config = Config._find_config_files(base_path)

    # Load model
    base_model = LlamaModel(config)
    load_model_weights(base_model, checkpoint, cache_dir=model_cache_dir, device=device)
    
    if args.method == 'lora':
        lora_model = LoraLlamaModel(base_model, args.lora_dim, args.lora_sigma)
        if os.path.exists('examples/ckpts/lora/lora_weights.pt'):
            print("LOAD OK")
            state_dict = torch.load('examples/ckpts/lora/lora_weights.pt')
            lora_model.model.load_state_dict(state_dict)
        
        model = LoRaClassifer(lora_model)
        if os.path.exists('./examples/ckpts/lora/class_weights.pt'):
            print("LOAD OK")
            class_state_dict = torch.load('./examples/ckpts/lora/class_weights.pt')
            model.load_state_dict(class_state_dict)
        
        print("USING LORA")
    else:
        model = LoRaClassifer(base_model)
        
    model = model.to(device)        
        
    max_num_sample = 500
        
    # Set up datasets and dataloaders
    train_n_val_dataset = CPEN455DatasetWrapper(csv_path=args.dataset_path)
    training_dataset, val_dataset = split_train_val(train_n_val_dataset, 0.8, return_remaining=True)
    test_dataset = CPEN455DatasetWrapper(csv_path=args.test_dataset_path)

    train_spam_num = len([i for i, (_, _, _, label) in enumerate(training_dataset) if label == 1])
    train_ham_num = len([i for i, (_, _, _, label) in enumerate(training_dataset) if label == 0])
    
    log_p_spam = torch.tensor((train_spam_num + 1)/(train_ham_num + train_spam_num+2)).log()
    log_p_ham = torch.tensor(((train_ham_num + 1)/(train_ham_num + train_spam_num+2))).log()

    print(f"P spam{torch.exp(log_p_ham)}, P spam {torch.exp(log_p_spam)}")

    training_dataloader = DataLoader(
        training_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
        )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
        )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
        )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-1)
        
    # Compute prior probability of spam/ham
    
    if os.path.exists(args.prob_output_folder) == False:
        os.makedirs(args.prob_output_folder)


    for iteration in tqdm(range(args.num_iterations), desc="Training Epoch"):
        # Evaluate Validation Loss per 10 steaps
        
        if (iteration + 1) % (args.num_iterations // 10) == 0:
            val_acc_logger = avg_acc_logger()
            val_loss_logger = avg_logger()
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Evaluating on validation set during training"):
                    
                    # bpd, is_correct, (probs, labels_pred) = train_or_test(
                    #     args = args, 
                    #     model = model, 
                    #     tokenizer = tokenizer, 
                    #     batch = batch, 
                    #     is_training=False)
                    
                    loss_val, is_correct, (probs, labels_pred) = discriminative_test(
                        args = args, 
                        model = model, 
                        tokenizer = tokenizer, 
                        batch = batch, 
                        is_training=False)
                    
                    val_acc_logger.update(is_correct)
                    val_loss_logger.update(loss_val.item())

                    wandb.log({
                        "val_avg_bpd": val_loss_logger.compute_average(),
                        "val_avg_accuracy": val_acc_logger.compute_accuracy(),
                        "training_iteration": iteration,
                        })
                    
        if not is_required_training(args.method):
            break
        
        # Reshuffle training dataloader
        
        for train_batch in training_dataloader:    
            # Train on this batch
            loss_val, is_correct, _ = discriminative_test(
                args = args, 
                model = model, 
                tokenizer = tokenizer, 
                optimizer= optimizer,
                batch = train_batch, 
                is_training=True,
                log_p_spam=log_p_spam,
                log_p_ham=log_p_ham )
            
        wandb.log({
            "training_batch_bpd": loss_val.item(),
            "training_batch_acc": is_correct.float().mean().item(),
            "training_iteration": iteration,
            })

    # After training, save probabilities on test set
    train_n_val_dataloader = DataLoader(
        train_n_val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
        )
    
    
    if args.method == 'lora':
        torch.save(model.model.model.state_dict(), 'examples/ckpts/lora/lora_weights.pt')
        torch.save(model.save_dict(), 'examples/ckpts/lora/class_weights.pt')
        
    save_probs(args, model, tokenizer, train_n_val_dataloader, device=device, name = "train_n_val", eval=True)
    save_probs(args, model, tokenizer, test_dataloader, device=device, name = "test", eval=False)
