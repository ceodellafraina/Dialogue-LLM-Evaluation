"""
This script caluclates the average 'overall' human scores for each dialogue in 'pc_usr_data.json' 
and saves them in another json for the comparison
"""
import json

with open('dataset/pc_usr_data.json', 'r') as file:
    data = json.load(file)

new_data = []

for entry in data:
    context = entry['context']
    for response_entry in entry['responses']:
        response = response_entry['response']
        model = response_entry['model']
        overall_scores = response_entry['Overall']
        avg_score = sum(overall_scores) / len(overall_scores) 
        new_data.append({
            "context": context.strip(),
            "response": response.strip(),
            "model": model,
            "score": avg_score
        })

with open('structured_dialogs.json', 'w') as outfile:
    json.dump(new_data, outfile, indent=4)

print("Success.")
