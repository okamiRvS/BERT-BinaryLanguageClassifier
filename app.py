# Load the libraries
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn # pip install uvicorn[standard]
import torch
import os
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model
name_model = "finetuning-binary-language-classifier"
rootPath = os.getcwd()
modelsPath = os.path.join(rootPath, "models")
output_dir = os.path.join(modelsPath, name_model)

# Import transformers
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained(output_dir)
model = BertForSequenceClassification.from_pretrained(output_dir)

from transformers import pipeline
from datasets import Dataset, DatasetDict
pipe = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=device)

# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to the Binary Language Classifier"}

class Item(BaseModel):
    text: str = Field(default=None, title="The description of the item", max_length=800)
    
# Define the route to the test
@app.post("/predict")
def predict(item: Item):

    input_arr = np.array([[
        item.text]])

    inDF = pd.DataFrame(input_arr, columns=["text"])
    dataForPrediction = Dataset.from_dict(inDF)["text"]

    tokenizer_kwargs = {
        #"padding":True,
        "truncation": True,
        "max_length": tokenizer.model_max_length, # if you want you can change, maybe
        "top_k": 1,
        "batch_size": 256,
        #return_tensors" : "pt"
    }

    prediction = []
    for out in tqdm(pipe(dataForPrediction, **tokenizer_kwargs), total=len(dataForPrediction)):
        prediction.append(out)

    mapTarget = {
        "LABEL_0" : 0,
        "LABEL_1" : 1
    }

    inDF["prediction"] = [mapTarget[pred[0]["label"]] for pred in prediction]
    inDF["score"] = [pred[0]["score"] for pred in prediction]
    print(inDF)
   
    return {
            "output": str(inDF["prediction"].values[0])
           }


if __name__ == "__main__":

    # REMARK
    # If you need to debug then launch the app.py from play button in VS
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # Otherwise if you want to start the API application on command line
    # to update always the application thanks to "reload" (you don't need to comment previous line) then:
    # uvicorn app:app --reload

    # DOCKER COMMAND
    # docker build -t fastapiapp:latest -f docker/Dockerfile .
    # docker run -p 5000:80 fastapiapp:latest
    # http://localhost:5000/docs