"""
For the evaluation of dialogues in the datasets, the following open source models were tested (set the variable MODEL_NAME with their names):

- chatglm3-6b-base
- meta-llama/Llama-2-13b-chat
- Qwen/Qwen-14B-Chat
- lmsys/vicuna-13b-v1.5
- baichuan-inc/Baichuan2-13B-Chat

Note: To test llama 2, you must request access to the model. For more information: https://huggingface.co/meta-llama/Llama-2-13b-chat
"""

""" The following prompts have been chosen based on the tendency of each model to understand the context and return the "Yes" or "No" token, 
following their training template: 


                                                    === VICUNA ===
https://ollama.com/library/vicuna:13b/blobs/b1571c5cbd28

PROMPT_TEMPLATE = 

USER: Here is a dialogue:

Dialogue: {context}
Response: {Response}

Is the response consistent with the dialogue? Reply with "Yes" or "No".
ASSISTANT:


                                                    === BAICHUAN ===
https://github.com/lm-sys/FastChat/issues/1958

PROMPT_TEMPLATE = 
Assistant:
User: Context: {context}
Response: {response}
Does this response follow the context correctly? Reply with "Yes" or "No".


                                                    === CHATGLM ===

PROMPT_TEMPLATE =
<|system|>
You are a helpful assistant that evaluates if a response is consistent with the context.
<|user|>
{context}\nResponse: {response}\nDoes this response follow the context correctly?"

<|assistant|>

                                                    === QWEN === 

PROMPT_TEMPLATE= 
<|im_start|>system
You are a helpful assistant that evaluates if a response is consistent with the context.
<|im_end|>
<|im_start|>user
Here is the context:

Context: {context}
Response: {response}

Is the response consistent with the context? Reply with "Yes" or "No".
<|im_end|>
<|im_start|>assistant


                                                        === LLAMA-2 ===

https://stackoverflow.com/questions/76772509/llama-2-7b-hf-repeats-context-of-question-directly-from-input-prompt-cuts-off-w
https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat
https://ollama.com/library/llama2:13b-chat/blobs/2e0493f67d0c

PROMPT_TEMPLATE =

[INST] <<SYS>>{{ .System }}<</SYS>>

{{ .Prompt }} [/INST]

PROMPT_TEMPLATE = 
<s>[INST] <<SYS>>
You are a model trained to evaluate whether a response follows the context of a conversation. Reply with "Yes" if the response follows the context, otherwise reply with "No."
<</SYS>>
Below is a dialogue. Evaluate if the response provided by the system is correct with respect to the context.

Context:
{context}

Response to evaluate: System: {response}
Does the response follow the context correctly? [/INST]

"""

import json
import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
"""from huggingface_hub import login

login()""" # Uncomment for login if you're using llama

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================
# CONFIGURATION
# ==============================
MODEL_NAME = "baichuan-inc/Baichuan2-13B-Chat" 
DATA_FILE = "datasets/fed_data.json" # If you use 'pc_usr_data.json' check the comments below in the 'processing data' section 
OUTPUT_FILE = "results/evaluation_results.json"

logging.info(f"Loading model {MODEL_NAME}...")

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Set the model to evaluation mode
model.eval()

logging.info("Model successfully loaded.")

# ==============================
# READING DATA
# ==============================
logging.info(f"Loading data from {DATA_FILE}...")
with open(DATA_FILE, 'r') as f:
    data = json.load(f)
logging.info(f"Data loaded: {len(data)} dialogues found.")

# ==============================
# PROMPT TEMPLATE
# ==============================
PROMPT_TEMPLATE = """
The given response:
{response}

Is relevant to the given context?
{context}

Yes or no?
"""

def generate_prompt(context, response):
    """
    Generate the prompt to provide to the model.
    """
    return PROMPT_TEMPLATE.format(context=context, response=response)

# ==============================
# SCORE CALCULATION
# ==============================
def calculate_overall_score(context, response):
    """
    Given the 'context' string and the 'response' string,
    perform a forward pass to obtain the logits of the last token
    and calculate the normalized probability of 'Yes' versus 'No'.
    """
    logging.info("Generating prompt...")
    prompt = generate_prompt(context, response)
    inputs = tokenizer(prompt, return_tensors="pt")

    # Get the token IDs for "Yes" and "No" (assuming they exist as single tokens)
    # If "Yes" or "No" are split into multiple subwords, consider handling differently.
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        logging.warning(
            "Warning: 'Yes' or 'No' are not single tokens for this tokenizer. "
            "Results might be inaccurate or inconsistent."
        )

    yes_token_id = yes_ids[0]
    no_token_id = no_ids[0]

    logging.info("Running the model to obtain logits...")
    with torch.no_grad():
        outputs = model(**inputs)
        # outputs.logits: [batch_size, sequence_length, vocab_size]
        logits = outputs.logits
        # Get the logits of the last token of the context
        last_token_logits = logits[0, -1, :]  # shape: [vocab_size]

        # Calculate probabilities using softmax
        probs = F.softmax(last_token_logits, dim=-1)

        p_yes = probs[yes_token_id].item()
        p_no = probs[no_token_id].item()

    logging.info(f"Calculated probabilities: P(Yes)={p_yes:.4f}, P(No)={p_no:.4f}")

    # Normalize the probability of "Yes"
    if (p_yes + p_no) > 0:
        normalized_p_yes = p_yes / (p_yes + p_no)
    else:
        normalized_p_yes = 0.0

    logging.info(f"Normalized score: P(Yes)={normalized_p_yes:.4f}")
    return normalized_p_yes

# ==============================
# PROCESSING DATA         
# ==============================
results = []
logging.info("Starting dialogue processing...")
for i, dialogue in enumerate(data):
    context = dialogue.get("context", "")
    response = dialogue.get("response", "")
    model_name = dialogue.get("system", "")
    logging.info(f"Processing dialogue {i+1}, response from model '{model_name}'...")
    score = calculate_overall_score(context, response)
    results.append({
        "context": context,
        "response": response,
        "model": model_name,
        "score": score
    })


# Use the above data processing script if you're using 'fed_data.json' dataset, 
# else if you use 'pc_usr_data.json':

"""
    results = []
logging.info("Starting dialogue processing...")
for i, dialogue in enumerate(data):
    context = dialogue.get("context", "")
    for response_data in dialogue.get("responses", []):
        response = response_data.get("response", "")
        model_name = response_data.get("model", "")
        logging.info(f"Processing dialogue {i+1}, response from model '{model_name}'...")
        score = calculate_overall_score(context, response)
        results.append({
            "context": context,
            "response": response,
            "model": model_name,
            "score": score
        })

"""

# ==============================
# SAVING RESULTS
# ==============================
logging.info(f"Saving results to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=4)
logging.info("Results successfully saved.")
