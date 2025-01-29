from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import GPT2Tokenizer, GPT2Model
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # -> replace with our frontend URL, I think. So Vercel or something
    allow_methods=["POST"], # -> a POST API requests sends information to the server, i.e. login-credentials
)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # -> get tokenizer which transforms text sequence in tokens
model = GPT2Model.from_pretrained('gpt2', output_attentions=True) # -> get model, which outputs attentions

@app.post("/process") # test with uvicorn main:app --reload :) then open http://localhost:8000/docs
async def process_text(text: str): # define process operation, i.e. what happens when a POST request has been sent
    try:
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # Run model
        outputs = model(**inputs) # ** notation is the unpack operator on dictionaries (confirm that inputs is a dictionary)
        attentions = outputs.attentions  # Tuple of attention tensors (layers x heads)

        # Convert tensors to lists
        attention_data = [
            layer_attention.tolist() for layer_attention in attentions
        ]

        return {
            "tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), # convert back to tokens
            "attention": attention_data
        }
    except Exception as e: # if post request failed
        raise HTTPException(status_code=500, detail=str(e))