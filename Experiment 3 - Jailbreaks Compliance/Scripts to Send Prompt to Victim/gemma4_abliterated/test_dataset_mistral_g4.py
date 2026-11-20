import pandas as pd
import ollama
import time

# =========================
# Configuration
# =========================

INPUT_FILE = "safire_g4.csv"

OUTPUT_RESULTS_FILE = "jailbreak_results_mistral_g4.csv"

TARGET_MODEL = "mistral:latest"

INTENSITIES = [0, -1, -2, -3, -4, -5, 1, 2, 3, 4, 5]

# =========================
# Verify Model
# =========================

print("Checking Ollama model...")

installed_models = ollama.list()

model_names = [
    model["model"]
    for model in installed_models["models"]
]

if TARGET_MODEL not in model_names:
    raise ValueError(
        f"Target model not installed: {TARGET_MODEL}"
    )

print("Model verified.\n")

# =========================
# Load Dataset
# =========================

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} prompts.\n")

# =========================
# Response Generation
# =========================

def generate_response(prompt):

    response = ollama.chat(
        model=TARGET_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()

# =========================
# Main Evaluation
# =========================

results = []

for idx, row in df.iterrows():

    print("=" * 60)
    print(f"PROMPT {idx + 1}/{len(df)}")
    print("=" * 60)

    result_row = {}

    # Store base prompt
    base_prompt = row["intensity_0"]

    result_row["prompt"] = base_prompt

    # Store reference refusal
    result_row["reference_response_refuse"] = row[
        "reference_response_refuse"
    ]

    # =========================
    # Generate responses
    # =========================

    for intensity in INTENSITIES:

        print(f"Testing intensity {intensity}...")

        intensity_column = f"intensity_{intensity}"

        prompt = row[intensity_column]

        try:

            start_time = time.time()

            model_response = generate_response(prompt)

            elapsed = time.time() - start_time

            print(
                f"  Response generated "
                f"({elapsed:.1f}s)"
            )

        except Exception as e:

            print(f"  Generation ERROR: {e}")

            model_response = "ERROR"

        response_column = f"response_{intensity}"

        result_row[response_column] = model_response

    results.append(result_row)

    # =========================
    # Save partial progress
    # =========================

    partial_df = pd.DataFrame(results)

    partial_df.to_csv(
        OUTPUT_RESULTS_FILE,
        index=False
    )

    print("\nPartial results saved.\n")

# =========================
# Save Final Results
# =========================

results_df = pd.DataFrame(results)

# Ensure exact column ordering
ordered_columns = [
    "prompt",
    "reference_response_refuse",
    "response_0",
    "response_-1",
    "response_-2",
    "response_-3",
    "response_-4",
    "response_-5",
    "response_1",
    "response_2",
    "response_3",
    "response_4",
    "response_5",
]

results_df = results_df[ordered_columns]

results_df.to_csv(
    OUTPUT_RESULTS_FILE,
    index=False
)

# =========================
# Final Print
# =========================

print("\nGeneration complete.\n")

print("Saved results:")
print(f"  {OUTPUT_RESULTS_FILE}")