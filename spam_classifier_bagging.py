#!/usr/bin/env python3

"""
Spam Classifier with Bagging Ensemble
Implements a bagging-based ensemble approach for email spam classification using LLMs.
Each model is trained on a bootstrap sample of the training data.
"""

import os
import torch
import wandb
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import DataLoader, RandomSampler
from torch.nn import functional as F

from autograder.dataset import CPEN455_2025_W1_Dataset, ENRON_LABEL_INDEX_MAP, prepare_subset
from model import LlamaModel
from utils.weight_utils import load_model_weights
from model.config import Config
from model.tokenizer import Tokenizer
from utils.download import _resolve_snapshot_path
from utils.device import set_device
from utils.prompt_template import get_prompt
from utils.logger import avg_logger, avg_acc_logger


def get_seq_log_prob(prompts, tokenizer, model, device):
    """
    Compute sequence log probabilities for a batch of prompts.
    
    Args:
        prompts: List of text prompts
        tokenizer: Tokenizer for encoding
        model: Language model
        device: Computation device
    
    Returns:
        Tensor of shape [batch_size] with log probability for each sequence
    """
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
    
    return gathered_log_prob.sum(dim=-1)


def train_one_iter(model, tokenizer, batch, optimizer, max_seq_length=256, 
                   is_training=True, device="cpu"):
    """
    Train/evaluate one iteration on a batch.
    
    Args:
        model: LLM model
        tokenizer: Tokenizer
        batch: Data batch (data_index, subjects, messages, labels)
        optimizer: Optimizer for gradient update
        max_seq_length: Maximum sequence length for prompts
        is_training: Whether in training mode
        device: Computation device
    
    Returns:
        Tuple of (loss, is_correct, (logits, predictions))
    """
    if is_training:
        model.train()
    else:
        model.eval()

    _, subjects, messages, label_indexs = batch
    
    if -1 in label_indexs:
        loss_val = None
        is_correct = None
    else:
        with torch.no_grad():
            spam_prompts = [
                get_prompt(subject=subj, message=msg, 
                          label=ENRON_LABEL_INDEX_MAP.inv[0], 
                          max_seq_length=max_seq_length) 
                for subj, msg in zip(subjects, messages)
            ]
            ham_prompts = [
                get_prompt(subject=subj, message=msg, 
                          label=ENRON_LABEL_INDEX_MAP.inv[1], 
                          max_seq_length=max_seq_length) 
                for subj, msg in zip(subjects, messages)
            ]

        spam_seq_log_prob = get_seq_log_prob(spam_prompts, tokenizer, model, device=device).cpu()
        ham_seq_log_prob = get_seq_log_prob(ham_prompts, tokenizer, model, device=device).cpu()
    
        softmax_logits = torch.stack((spam_seq_log_prob, ham_seq_log_prob), dim=1)

        ce_loss = torch.nn.CrossEntropyLoss()
        loss_val = ce_loss(softmax_logits, label_indexs)

        if is_training: 
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
        labels_pred = torch.argmax(softmax_logits, dim=-1)
        is_correct = labels_pred == label_indexs
        
    return loss_val, is_correct, softmax_logits.detach().cpu() if 'softmax_logits' in locals() else None


class SpamClassifier:
    """
    Bagging-based ensemble classifier for spam detection using LLMs.
    
    Trains multiple models on bootstrap samples of the data and combines
    their predictions through voting.
    """
    
    def __init__(self, n_models, config, train_val_dataset, 
                 checkpoint, model_cache_dir, device="cpu"):
        """
        Initialize the spam classifier ensemble.
        
        Args:
            n_models: Number of models in the ensemble
            config: Model configuration
            train_val_dataset: Training/validation dataset
            checkpoint: Model checkpoint path
            model_cache_dir: Cache directory for model weights
            device: Computation device
        """
        self.config = config    
        self.n_models = n_models
        self.models = [LlamaModel(config).to(device) for _ in range(n_models)]
        
        # Load pre-trained weights for all models
        for model in self.models:
            load_model_weights(model, checkpoint, model_cache_dir, device=device)            
        
        # Split into train/val (80/20)
        training_dataset, val_dataset = prepare_subset(
            train_val_dataset, 
            int(0.8 * len(train_val_dataset)), 
            ratio_spam=0.5, 
            return_remaining=True
        )
        
        self.training_dataset = training_dataset
        self.val_dataset = val_dataset
        self.device = device
        
        # Create optimizer for each model
        self.optimizers = [
            torch.optim.AdamW(model.parameters(), lr=1e-5) 
            for model in self.models
        ]
  
    def predict(self, tokenizer, batch, max_seq_length=256, 
                return_loss=False):
        """
        Make predictions using ensemble voting.
        
        Args:
            tokenizer: Tokenizer
            batch: Data batch
            max_seq_length: Maximum sequence length
            return_loss: Whether to return validation loss
        
        Returns:
            Predictions (and loss if return_loss=True)
        """
        model_predictions = []
        avg_loss = 0.0        
    
        with torch.no_grad():
            for model in self.models:
                loss_val, _, _ = train_one_iter(
                    model, tokenizer, batch, 
                    optimizer=None,
                    max_seq_length=max_seq_length,
                    is_training=False, 
                    device=self.device
                )
                
                if return_loss and loss_val is not None:
                    avg_loss += loss_val.item()
                
                # Get predictions from this model
                _, subjects, messages, labels = batch
                spam_prompts = [
                    get_prompt(subject=subj, message=msg, 
                              label=ENRON_LABEL_INDEX_MAP.inv[0], 
                              max_seq_length=max_seq_length) 
                    for subj, msg in zip(subjects, messages)
                ]
                ham_prompts = [
                    get_prompt(subject=subj, message=msg, 
                              label=ENRON_LABEL_INDEX_MAP.inv[1], 
                              max_seq_length=max_seq_length) 
                    for subj, msg in zip(subjects, messages)
                ]
                
                spam_log_prob = get_seq_log_prob(spam_prompts, tokenizer, model, self.device).cpu()
                ham_log_prob = get_seq_log_prob(ham_prompts, tokenizer, model, self.device).cpu()
                
                logits = torch.stack((spam_log_prob, ham_log_prob), dim=1)
                pred = torch.argmax(logits, dim=-1)
                model_predictions.append(pred)
            
            avg_loss = avg_loss / self.n_models if return_loss else 0
            
            # Majority voting
            model_predictions = torch.stack(model_predictions)  # [n_models, batch_size]
            voted_pred = torch.mode(model_predictions, dim=0).values
        
            if return_loss:
                return voted_pred, avg_loss
            else:
                return voted_pred

    def _evaluate_validation(self, val_dataloader, tokenizer,
                            max_seq_length=256) -> Tuple[float, float]:
        """
        Evaluate on validation set.
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        val_loss_logger = avg_logger()
        val_accuracy_logger = avg_acc_logger()
        
        for val_batch in val_dataloader:
            _, _, _, label_index = val_batch 
            model_predictions, avg_loss = self.predict(
                tokenizer, val_batch, 
                max_seq_length, return_loss=True
            )
            
            if avg_loss is not None:
                val_loss_logger.update(avg_loss)
            
            is_correct = model_predictions == label_index
            val_accuracy_logger.update(is_correct)
        
        return val_loss_logger.compute_average(), val_accuracy_logger.compute_accuracy()
        
    def train(self, tokenizer, num_iterations, batch_size, max_seq_length=256):
        """
        Train all models with bagging (bootstrap sampling).
        
        Each model is trained on a bootstrap sample of the training data,
        creating diversity in the ensemble.
        
        Args:
            tokenizer: Tokenizer
            num_iterations: Number of training iterations
            batch_size: Batch size for training
            max_seq_length: Maximum sequence length for prompts
        """
        # Create bootstrap sampler
        sampler = RandomSampler(
            self.training_dataset, 
            replacement=True, 
            num_samples=len(self.training_dataset)
        )
                
        train_loader = DataLoader(
            self.training_dataset, 
            batch_size=batch_size, 
            sampler=sampler
        )
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size,
            shuffle=False
        )
                    
        for iteration in tqdm(range(num_iterations), desc="Training"):
            # Validate every 10% of iterations
            if (iteration + 1) % max(1, num_iterations // 10) == 0:
                with torch.no_grad():
                    avg_loss, avg_acc = self._evaluate_validation(
                        val_loader, tokenizer, max_seq_length
                    )
                    
                    wandb.log({
                        'val_avg_loss': avg_loss,
                        'val_avg_acc': avg_acc,
                        'training_iteration': iteration
                    })
            
            train_loss_logger = avg_logger()
            train_acc_logger = avg_acc_logger()
            
            # Train each model independently with bootstrap samples
            for model_idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                train_batch = next(iter(train_loader))
                                
                loss_val, is_correct, _ = train_one_iter(
                    model, tokenizer, train_batch, optimizer,
                    max_seq_length=max_seq_length,
                    is_training=True,
                    device=self.device
                )
                
                if loss_val is not None:
                    train_loss_logger.update(loss_val.item())
                if is_correct is not None:
                    train_acc_logger.update(is_correct)
    
            wandb.log({
                'train_avg_loss': train_loss_logger.compute_average(),
                'train_avg_acc': train_acc_logger.compute_accuracy(), 
                'train_iteration': iteration
            })




def save_test_probs(classifier: SpamClassifier, tokenizer, test_dataloader, name="test", max_seq_length=256):
    save_path = os.path.join(os.getcwd(), f"{name}_dataset_probs.csv")

    # Remove file if it exists
    if os.path.exists(save_path):
        os.remove(save_path)

    # Write header once
    with open(save_path, "w", newline="") as f:
        f.write("data_index,prob_ham,prob_spam\n")

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="saving probabilities"):
            # Predicted labels from classifier
            pred_labels = classifier.predict(tokenizer, batch, max_seq_length, return_loss=False)

            # batch[3] = indices (based on your code earlier)
            _, _, _, data_indices = batch  

            rows = []
            for idx, pred in zip(data_indices.tolist(), pred_labels.tolist()):
                if pred == 0:  # assume 0 = ham, 1 = spam
                    prob_ham, prob_spam = 1.0, 0.0
                else:
                    prob_ham, prob_spam = 0.0, 1.0

                rows.append(f"{idx},{prob_ham},{prob_spam}\n")

            # Append to CSV
            with open(save_path, "a", newline="") as f:
                f.writelines(rows)


def main():
    """Main training script."""
    from dotenv import load_dotenv
    
    # Configuration
    batch_size = 8
    max_seq_len = 256
    dataset_path = "autograder/cpen455_released_datasets/train_val_subset.csv"
    test_path = "autograder/cpen455_released_datasets/test_subset.csv"
    num_iterations = 100
    num_models = 10
    
    # Load environment
    load_dotenv()
    checkpoint = os.getenv("MODEL_CHECKPOINT")
    model_cache_dir = os.getenv("MODEL_CACHE_DIR")
    
    # Initialize wandb
    run = wandb.init(
        project=os.getenv("PROJECT_NAME"),
        name=f"spam-classifier-bagging_msl{max_seq_len}_ni{num_iterations}_nm{num_models}",
    )
    
    # Set device
    device = set_device()
    
    # Load components
    tokenizer = Tokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir)
    base_path = _resolve_snapshot_path(checkpoint, cache_dir=model_cache_dir)
    config = Config._find_config_files(base_path)
    
    # Load dataset
    train_val_dataset = CPEN455_2025_W1_Dataset(csv_path=dataset_path)
    
    # Create and train classifier
    classifier = SpamClassifier(
        num_models, config, train_val_dataset, 
        checkpoint, model_cache_dir, device=device
    )
    
    classifier.train(tokenizer, num_iterations, batch_size, max_seq_len)
    
    # Predict test dataset
    test_dataset = CPEN455_2025_W1_Dataset(csv_path=test_path)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False
    )
    
    save_test_probs(classifier, tokenizer, test_loader)
    
    wandb.finish()


if __name__ == "__main__":
    main()
