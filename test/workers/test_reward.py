def test_forward():
    from workers.reward import Reward
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Initialize the reward model
    model_name = "NotTheStallion/BERT-RM"
    num_layers = 5
    reward_model = Reward(model_name, num_layers)

    # Create a dummy input
    input_text = "This is a test input."
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Forward pass through the model
    output = reward_model(inputs["input_ids"], attention_mask=inputs["attention_mask"])

    # Check the output shape
    assert output.last_hidden_state.shape == (1, inputs["input_ids"].shape[1], 768), "Output shape mismatch"

def main():
    # Run the test
    test_forward()
    print("Test passed!")

if __name__ == "__main__":
    main()