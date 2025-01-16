import json
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, MinLengthLogitsProcessor
import torch
import torch.nn.functional as F
import logging

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurazione modello
MODEL_NAME = "THUDM/chatglm3-6b-base"
logging.info(f"Caricamento del modello {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
logging.info("Modello caricato con successo.")

# Carica il file JSON
DATA_FILE = "dataset/fed_data.json" 
logging.info(f"Caricamento dei dati da {DATA_FILE}...")
with open(DATA_FILE, 'r') as f:
    data = json.load(f)
logging.info(f"Dati caricati: {len(data)} dialoghi trovati.")

# Funzione per generare il prompt (dal file prompt_opensource.png)
PROMPT_TEMPLATE = """
### Context:
{context}

### Response:
{response}

### Instruction:
Does the response appropriately and effectively address the context? Answer with "Yes" or "No" only.
"""

def generate_prompt(context, response):
    return PROMPT_TEMPLATE.format(context=context, response=response)

# Funzione per calcolare le probabilità normalizzate di "Yes" e "No"
def calculate_overall_score(context, response):
    logging.info("Generazione del prompt...")
    prompt = generate_prompt(context, response)
    inputs = tokenizer(prompt, return_tensors="pt")
    logging.info("Esecuzione del modello per generare il punteggio...")
    with torch.no_grad():
        # Limitare il vocabolario ai token "Yes" e "No"
        yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

        logits_processor = LogitsProcessorList([
            MinLengthLogitsProcessor(1, eos_token_id=yes_token_id)  # Forza almeno un token con "Yes" come EOS
        ])

        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            logits_processor=logits_processor,
            output_scores=True,
            return_dict_in_generate=True
        )

        # Decodifica del testo generato
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip()
        logging.info(f"Testo generato: {generated_text}")

        # Calcolo delle probabilità
        scores = outputs.scores[-1]
        probs = F.softmax(scores, dim=-1)

        p_yes = probs[0, yes_token_id].item()
        p_no = probs[0, no_token_id].item()

        logging.info(f"Probabilità calcolate: P(Yes)={p_yes:.4f}, P(No)={p_no:.4f}")

        # Normalizzazione della probabilità di "Yes"
        if p_yes + p_no > 0:
            normalized_p_yes = p_yes / (p_yes + p_no)
        else:
            normalized_p_yes = 0.0

        logging.info(f"Punteggio normalizzato: P(Yes)={normalized_p_yes:.4f}")

    return normalized_p_yes

# Elaborazione dei dati e calcolo dei punteggi
results = []
logging.info("Inizio elaborazione dei dialoghi...")
for i, dialogue in enumerate(data):
    context = dialogue.get("context", "")
    response = dialogue.get("response", "")
    model_name = dialogue.get("system", "")
    logging.info(f"Elaborazione dialogo {i+1}, risposta del modello '{model_name}'...")
    score = calculate_overall_score(context, response)
    results.append({
        "context": context,
        "response": response,
        "model": model_name,
        "score": score
    })

# Salva i risultati
OUTPUT_FILE = "results/evaluation_results_chatglm3-6b-base_fed.json"
logging.info(f"Salvataggio dei risultati in {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=4)
logging.info("Risultati salvati con successo.")
