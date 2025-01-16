import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import logging

# Verifica e settaggio del device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# Nome del modello
MODEL_NAME = "lmsys/vicuna-13b-v1.5"

logging.info(f"Caricamento del modello {MODEL_NAME}...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    device_map="auto",            # Distribuisce i pesi in automatico (se la GPU non basta)
    offload_folder="./offload"    # Cartella di offload per pesi troppo grandi
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

logging.info("Modello caricato con successo.")

# Passiamo il modello in modalità valutazione (non training)
model.eval()

# Carica il file JSON
DATA_FILE = "dataset/fed_data.json"
logging.info(f"Caricamento dei dati da {DATA_FILE}...")
with open(DATA_FILE, 'r') as f:
    data = json.load(f)
logging.info(f"Dati caricati: {len(data)} dialoghi trovati.")

# Template per costruire il prompt
PROMPT_TEMPLATE = """
### Context:
{context}

### Response:
{response}

### Instruction:
Does the response appropriately and effectively address the context? Answer with "Yes" or "No" only.
"""

def generate_prompt(context, response):
    """ Crea il prompt da dare in input al modello """
    return PROMPT_TEMPLATE.format(context=context, response=response)

class OnlyYesNoProcessor(LogitsProcessor):
    """
    Forza il modello a generare solo i token corrispondenti a "Yes" o "No".
    Tutti gli altri token avranno punteggio -inf (praticamente impossibile da generare).
    """
    def __init__(self, yes_id, no_id):
        super().__init__()
        self.yes_id = yes_id
        self.no_id = no_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Imposta tutti i token a -inf, tranne quelli di "Yes" e "No"
        new_scores = torch.full_like(scores, float('-inf'))
        new_scores[:, self.yes_id] = scores[:, self.yes_id]
        new_scores[:, self.no_id] = scores[:, self.no_id]
        return new_scores

def calculate_overall_score(context, response):
    """
    1) Crea il prompt.
    2) Tokenizza il prompt.
    3) Esegue una generazione di 1 token con un LogitsProcessor che permette SOLO 'Yes' o 'No'.
    4) Calcola la probabilità normalizzata di 'Yes'.
    5) Logga anche la risposta generata.
    """
    logging.info("Generazione del prompt...")
    prompt = generate_prompt(context, response)
    
    # Tokenizza
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    logging.info("Esecuzione del modello per generare il punteggio...")
    # Otteniamo gli ID dei token "Yes" e "No"
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

    # Creiamo il nostro LogitsProcessor personalizzato
    only_yes_no_processor = OnlyYesNoProcessor(yes_token_id, no_token_id)

    with torch.no_grad():
        # Generiamo un solo token (Yes o No)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
            logits_processor=LogitsProcessorList([only_yes_no_processor]),
            # Per avere un comportamento deterministico:
            # - disabilitiamo il campionamento (do_sample=False)
            # - temperature = 0.0
            do_sample=False,
            temperature=0.0
        )

        # Decodifichiamo il token appena generato (rispetto alla lunghezza del prompt)
        generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]  
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        logging.info(f"Risposta generata (token unico): {generated_text}")

        # Recuperiamo il vettore di punteggi dell'ultimo token
        last_scores = outputs.scores[-1][0]  # shape [vocab_size]
        probs = F.softmax(last_scores, dim=-1)

        p_yes = probs[yes_token_id].item()
        p_no = probs[no_token_id].item()

        logging.info(f"Probabilità calcolate: P(Yes)={p_yes:.4f}, P(No)={p_no:.4f}")

        # Normalizzazione
        if (p_yes + p_no) > 0:
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

# Salvataggio dei risultati
OUTPUT_FILE = "results/evaluation_results.json"
logging.info(f"Salvataggio dei risultati in {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=4)

logging.info("Risultati salvati con successo.")
