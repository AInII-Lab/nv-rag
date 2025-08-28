import random
import os

import pandas as pd

import tqdm

from dotenv import load_dotenv

load_dotenv()

from together import Together

client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

CHUNKS_PATH = "xxx"
SAVE_TO = "xxx"


def get_prompt(chunk):
    return f"""
    Generate a question in German that can be answered with the following text chunk. Answer only with the question in German, nothing else.
    
    Chunk:
    {chunk}
    
    Question:
    """


def get_question(chunk):
    prompt = get_prompt(chunk)
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>"],
        stream=False,
    )
    return response.choices[0].message.content


# read in chunks (from .csv or whatever)
chunks = pd.read_csv(CHUNKS_PATH)

random_chunks = chunks.sample(305)

questions = []

for i, chunk in tqdm.tqdm(random_chunks.iterrows(), total=len(random_chunks)):
    q = get_question(chunk["text"])
    questions.append(q)

random_chunks["example_questions"] = questions

random_chunks.to_csv(SAVE_TO, index=False)
print(f"Saved the example questions to '{SAVE_TO}'!")
