import json
import logging
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================
# CONFIGURAZIONE
# ==============================
MODEL_NAME = "baichuan-inc/Baichuan2-13B-Chat"
DATA_FILE = "dataset/fed_data.json"
OUTPUT_FILE = "results/evaluation_results.json"

logging.info(f"Caricamento del modello {MODEL_NAME}...")

# Caricamento del modello e del tokenizer in modo "generico"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Metti il modello in evaluation mode
model.eval()

logging.info("Modello caricato con successo.")

# ==============================
# LETTURA DEI DATI
# ==============================
logging.info(f"Caricamento dei dati da {DATA_FILE}...")
with open(DATA_FILE, 'r') as f:
    data = json.load(f)
logging.info(f"Dati caricati: {len(data)} dialoghi trovati.")

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
    Genera il prompt da fornire al modello.
    """
    return PROMPT_TEMPLATE.format(context=context, response=response)

# ==============================
# CALCOLO DEL PUNTEGGIO
# ==============================
def calculate_overall_score(context, response):
    """
    Data la stringa 'context' e la stringa 'response',
    esegue un forward pass per ottenere i logits
    dell'ultimo token e calcolare la probabilità normalizzata
    di 'Yes' rispetto a 'No'.
    """
    logging.info("Generazione del prompt...")
    prompt = generate_prompt(context, response)
    inputs = tokenizer(prompt, return_tensors="pt")

    # Ottieni gli ID dei token "Yes" e "No" (assumendo che esistano come singoli token)
    # Se "Yes" o "No" fossero spezzati in più subword, considera di gestire la cosa diversamente.
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        logging.warning(
            "Attenzione: 'Yes' o 'No' non sono singoli token per questo tokenizer. "
            "I risultati potrebbero non essere corretti o coerenti."
        )

    yes_token_id = yes_ids[0]
    no_token_id = no_ids[0]

    logging.info("Esecuzione del modello per ottenere i logits...")
    with torch.no_grad():
        outputs = model(**inputs)
        # outputs.logits: [batch_size, sequence_length, vocab_size]
        logits = outputs.logits
        # Otteniamo i logits dell'ultimo token del contesto
        last_token_logits = logits[0, -1, :]  # shape: [vocab_size]

        # Calcoliamo le probabilità con una softmax
        probs = F.softmax(last_token_logits, dim=-1)

        p_yes = probs[yes_token_id].item()
        p_no = probs[no_token_id].item()

    logging.info(f"Probabilità calcolate: P(Yes)={p_yes:.4f}, P(No)={p_no:.4f}")

    # Normalizzazione della probabilità di "Yes"
    if (p_yes + p_no) > 0:
        normalized_p_yes = p_yes / (p_yes + p_no)
    else:
        normalized_p_yes = 0.0

    logging.info(f"Punteggio normalizzato: P(Yes)={normalized_p_yes:.4f}")
    return normalized_p_yes

# ==============================
# ELABORAZIONE DEI DATI
# ==============================
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

# ==============================
# SALVATAGGIO DEI RISULTATI
# ==============================
logging.info(f"Salvataggio dei risultati in {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=4)
logging.info("Risultati salvati con successo.")