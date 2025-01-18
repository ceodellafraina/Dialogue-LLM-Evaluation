"""
This script caluclates the average 'overall' human scores for each dialogue in 'fed_data.json' 
and saves them in another json for the comparison
"""

import json

with open('dataset/fed_data.json', 'r') as file:
    data = json.load(file)


new_data = []

for entry in data:
    context = entry.get('context', '').strip()
    response = entry.get('response', '').strip()  # if there's no 'response' field
    model = entry.get('system', 'Unknown')
    annotations = entry.get('annotations', {})
    overall_scores = annotations.get('Overall', [])

    
    avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0


    new_data.append({
        "context": context,
        "response": response,
        "model": model,
        "score": avg_score
    })


with open('structured_fed_data.json', 'w') as outfile:
    json.dump(new_data, outfile, indent=4)

print("Il nuovo JSON è stato creato con successo.")
