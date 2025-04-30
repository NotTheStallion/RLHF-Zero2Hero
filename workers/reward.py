import torch
from transformers import AutoModel
from datasets import load_dataset
from torch.utils.data import DataLoader
# Removed unused import
from transformers import AutoTokenizer
import wandb
from huggingface_hub import Repository

# Removed unused import

import torch.nn as nn

class Reward(nn.Module):
    def __init__(self, transformer_model_name: str, num_transformer_layers: int = 5):
        super(Reward, self).__init__()

        # Try to load a pre-trained transformer model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        self.model = AutoModel.from_pretrained(transformer_model_name, trust_remote_code=True)

    def forward(self, input_ids, attention_mask=None):
        reward = self.model(input_ids, attention_mask=attention_mask)
        return reward

    def train_reward(self, batch, optimizer):
        optimizer.zero_grad()
        
        input_ids, chosen_response, rejected_response, _, _ = batch
        
        # The reward model takes as an input the concatenation of the input and response
        max_length = self.tokenizer.model_max_length
        chosen_inputs = torch.cat((input_ids, chosen_response), dim=1)[:, :max_length]
        rejected_inputs = torch.cat((input_ids, rejected_response), dim=1)[:, :max_length]
        
        # Compute the reward scores for chosen and rejected responses
        chosen_attention_mask = torch.ones_like(chosen_inputs)
        rejected_attention_mask = torch.ones_like(rejected_inputs)
        
        chosen_reward = self(chosen_inputs, chosen_attention_mask)
        rejected_reward = self(rejected_inputs, rejected_attention_mask)
        
        # print("Chosen reward:", chosen_reward)
        # print("Rejected reward:", rejected_reward)
        # print(f"=== c-r = {chosen_reward.item()-rejected_reward.item()} ===")
        
        # Compute loss -E(log(sigmoid(r(x,y_chosen)-r(x,y_rejected)))
        # where r(x,y) is the reward score for input x and response y
        
        # We know that chosen_reward > rejected_reward if the model predicts otherwise
        # we want to penalize it
        loss = -torch.mean(
            torch.log(torch.sigmoid( chosen_reward - rejected_reward )) 
        )
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        return loss.item()


if __name__ == "__main__":
    model_name = "NotTheStallion/Qwen2-Reward-Model-0.5B"
    num_layers = 5
    reward_model = Reward(model_name, num_layers)
    
    print(reward_model.model)
    
    
    # input = "Example input text."
    # tokenized_input = reward_model.tokenizer(input, return_tensors="pt")
    # input_ids = tokenized_input["input_ids"]
    # attention_mask = tokenized_input["attention_mask"]
    
    # print("Input IDs:", input_ids)
    # print("Attention Mask:", attention_mask)
    
    # reward_score = reward_model(input_ids, attention_mask)
    # print("Reward score:", reward_score)
    
    
    
    # print(reward_model)
    
    exit()
    
    data_name = "argilla/ultrafeedback-binarized-preferences"
    
    def load_and_split_data(data_name, split_ratio=0.8, max_size=None):
        dataset = load_dataset(data_name)
        
        # Optionally limit the size of the dataset
        if max_size is not None:
            dataset['train'] = dataset['train'].select(range(min(max_size, len(dataset['train']))))
        
        train_size = int(split_ratio * len(dataset['train']))
        val_size = len(dataset['train']) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset['train'], [train_size, val_size])
        return train_dataset, val_dataset

    def create_dataloader(dataset, batch_size=1):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load and split the dataset
    train_dataset, val_dataset = load_and_split_data(data_name, max_size=100)

    # Create dataloaders
    train_dataloader = create_dataloader(train_dataset, batch_size=1)
    val_dataloader = create_dataloader(val_dataset, batch_size=1)
    
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-5)
    
    exit()

    wandb.init(project="rlhf", name="reward_model_overfit")
    for batch in train_dataloader:
        # Testing overfittting
        for _ in range(1000):
            # Unpack the batch
            input_ids = batch['instruction']
            chosen_response = batch['chosen_response']
            rejected_response = batch['rejected_response']
            chosen_avg_rating = batch['chosen_avg_rating']
            rejected_avg_rating = batch['rejected_avg_rating']
            
            # Tokenize inputs and responses
            input_ids = reward_model.tokenizer(input_ids, return_tensors="pt", padding=True, truncation=True)["input_ids"]
            chosen_response = reward_model.tokenizer(chosen_response, return_tensors="pt", padding=True, truncation=True)["input_ids"]
            rejected_response = reward_model.tokenizer(rejected_response, return_tensors="pt", padding=True, truncation=True)["input_ids"]
            
            # Train the reward model
            loss = reward_model.train_reward((input_ids, chosen_response, rejected_response, chosen_avg_rating, rejected_avg_rating), optimizer)
            
            # Log the loss and other information to wandb
            wandb.log({"loss": loss})
        break
    
    # Finish the wandb run
    wandb.finish()
    
    