import json
import os
import time
from openai import OpenAI

# OpenAi Client
client = OpenAI(api_key="YOUR_TOKEN", base_url="YOUR_ENDPOINT")

# I/O
input_file = "datasets/pc_usr_avg.json"  # Path to the uploaded file
output_file = "pc_gpt4.json"

with open(input_file, "r") as f:
    data = json.load(f)

# Load previously processed data if available
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        processed_data = json.load(f)
else:
    processed_data = []

# Obtain already processed IDs
processed_ids = {record.get("context") + record.get("response") for record in processed_data}

# Evaluation prompt
PROMPT = """
### Context:
{context}
### Response:
{response}
### Instruction:
Rate the overall quality of the response on a scale of 1 to 5 and just output the corresponding ratings.
### Output Format: x
### Your Response: [Here is GPT-4’s output]
"""

# Iterate over unprocessed records
for idx, record in enumerate(data):
    context = record.get("context", "")
    response = record.get("response", "")
    unique_id = context + response  # Unique key to avoid duplicates

    if unique_id in processed_ids:
        continue  # Skip already processed records

    # Generate the prompt for OpenAI
    prompt = PROMPT.format(context=context, response=response)

    try:
        # API call to the model
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        score = result.choices[0].message.content.strip()  # Extract the score

        # Update the record with the score
        record["score"] = float(score)

        # Add the processed record to the data
        processed_data.append(record)

        # Save the processed data to the output file
        with open(output_file, "w") as f:
            json.dump(processed_data, f, indent=4)

        # Log progress
        print(f"Record {idx + 1}/{len(data)} processed and saved.")

    except Exception as e:
        print(f"Error processing record {idx + 1}: {e}")

        # Pause if a rate limit error occurs
        if "rate limit" in str(e).lower():
            print("Rate limit reached. Waiting for 1 minute.")
            time.sleep(60)

    # Add a delay between requests to avoid rate limits
    time.sleep(15)

print("Processing completed.")
