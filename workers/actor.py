import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



import os

class Actor:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.getenv("CACHE_DIR", None))
        self.eos_token = self.tokenizer.eos_token
        self.pad_token = self.tokenizer.pad_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=os.getenv("CACHE_DIR", None))

        self.model.config.pad_token_id = self.model.config.pad_token_id or self.model.config.eos_token_id
    
    def __call__(self, *args, **xargs) -> torch.Tensor:
        outputs = self.model.generate(*args, **xargs)
        return outputs
        
    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def inference(self, prompt: str, max_length: int = 50) -> str:
        inputs = self.encode(prompt)
        outputs = self.model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=max_length)
        return self.tokenizer.decode(outputs[0])

    def rollout(self, prompts: list, max_length: int = 50) -> list:
        responses = []
        for prompt in prompts:
            response = self.inference(prompt, max_length)
            responses.append(response)
        return responses


if __name__ == "__main__":
    actor = Actor("NotTheStallion/Qwen2.5-0.17B-layer-reduced")
    # prompts = ["Once upon a time", "In a galaxy far, far away"]
    # responses = actor.rollout(prompts)
    # for prompt, response in zip(prompts, responses):
    #     print(f"= Prompt: {prompt}\n= Response: {response}\n")
    
    # Tokenize a single character and print its shape
    # single_char = "random phrase"
    # encoded_char = actor.encode(single_char)
    # print("Encoded single character shape:", encoded_char["input_ids"])
    
    # Test call method with encode decode
    test_text = "Once upon a time,"
    encoded_text = actor.encode(test_text)
    print("Encoded text shape:", encoded_text["input_ids"].shape)
    output = actor(input_ids=encoded_text["input_ids"], attention_mask=encoded_text["attention_mask"], max_new_tokens=10)
    print("Ouput shape:", output.shape)
    decoded_text = actor.decode(output[0])
    # print("Decoded text:", decoded_text)