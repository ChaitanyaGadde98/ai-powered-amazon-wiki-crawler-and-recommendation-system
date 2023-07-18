from transformers import AutoTokenizer, AutoModel
import torch


def get_text_vector(text):
    # Load the pre-trained transformer model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the text
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    # Pass the tokenized input through the model to get the vector representation
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Extract the vector representation from the model's output
    vector = model_output.pooler_output.squeeze(0).numpy()

    return vector
