import json
import os
import time
from openai import OpenAI

# OpenAi Client
client = OpenAI(api_key="YOUR_TOKEN", base_url="YOUR_ENDPOINT")

# I/O
input_file = "datasets/fed_data_avg.json"
output_file = "pc_gpt4.json"


with open(input_file, "r") as f:
    data = json.load(f)

# Carica i dati già elaborati (se presenti)
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        processed_data = json.load(f)
else:
    processed_data = []

# Ottieni gli ID già elaborati
processed_ids = {record.get("context") + record.get("response") for record in processed_data}

# Prompt di valutazione
PROMPT = """
### Dialogues:
{context}
{response}
## Instruction:
Rate the overall quality of the input dialogue on a scale of 1 to 5 and just output the corresponding ratings.
### Output Format: x
### Your Response: [Here is GPT-4’s output]
"""

# Itera sui record non elaborati
for idx, record in enumerate(data):
    context = record.get("context", "")
    response = record.get("response", "")
    unique_id = context + response  # Chiave unica per evitare duplicati

    if unique_id in processed_ids:
        continue  # Salta i record già elaborati

    # Genera il prompt per OpenAI
    prompt = PROMPT.format(context=context, response=response)

    try:
        # Chiamata al modello
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        score = result.choices[0].message.content.strip()  # Estrarre il punteggio

        # Aggiorna il record con il punteggio
        record["score"] = float(score)

        # Aggiungi il record elaborato ai dati processati
        processed_data.append(record)

        # Salva i dati elaborati nel file di output
        with open(output_file, "w") as f:
            json.dump(processed_data, f, indent=4)

        # Log dell'avanzamento
        print(f"Record {idx + 1}/{len(data)} elaborato e salvato.")

    except Exception as e:
        print(f"Errore durante l'elaborazione del record {idx + 1}: {e}")

        # Pausa in caso di errore per limite di richieste
        if "rate limit" in str(e).lower():
            print("Raggiunto il limite di richieste. Attendere 1 minuto.")
            time.sleep(60)

    # Aggiungi un ritardo tra ogni richiesta per evitare problemi di limite
    time.sleep(15)

print("Elaborazione completata.")