import json
import os
import time

def evaluate_pipeline():
    """
    Evaluation set loader.
    The actual `ragas` metrics execution (faithfulness) and pipeline routing
    is handled downstream in `main.py`'s `/evaluate` endpoint to avoid
    circular imports. This function just safely loads the dataset.
    """
    eval_path = "/app/data/eval_set/questions.json"
    if not os.path.exists(eval_path):
        eval_path = "./data/eval_set/questions.json"
        
    try:
        with open(eval_path, "r") as f:
            questions = json.load(f)
    except FileNotFoundError:
        return {"error": "questions.json not found"}
        
    # We would normally import the query pipeline here, but since this
    # is called from main.py which has the pipeline, we can just return
    # the dataset to be processed in the route, or do an HTTP request to ourselves.
    
    # We will let main.py handle the pipeline calling to avoid circular imports.
    return questions
