"""Dataset generation and loading utilities."""

import os
import random
import numpy as np
import google.generativeai as genai


GEMINI_PROMPT = """
I am building a dataset for training a compression algorithm for IoT sensor networks (SSN/SOSA ontology). I need you to generate diverse natural language templates for sensor descriptions.

Please follow these strict rules:
1. Do NOT include specific numbers or IDs. Use placeholders: <ID>, <VALUE>, <UNIT>, <SENSOR_TYPE>, <LOCATION>.
2. Based on these seed examples, generate 50 distinct variations for each category using different vocabulary (synonyms like "device", "unit", "detector" instead of just "sensor").

Category A: Device Descriptions
Seed: "rangefinder #30 is a laser range finder sensor."
Target Output: "<SENSOR_TYPE> <ID> functions as a precision <SENSOR_TYPE> unit."

Category B: Observation Descriptions
Seed: "the height of tree #124"
Target Output: "measured <PROPERTY> of <OBJECT> <ID>"

Category C: Status Messages
Seed: "battery level is low"
Target Output: "Critical power failure on <SENSOR_TYPE> <ID>"

Generate 50 templates for each category (150 total lines). Output ONLY the raw text lines, no numbering or markdown.
"""


def fetch_templates_from_gemini(template_file, api_key):
    """Fetch synthetic templates from Gemini API or load from cache."""
    if os.path.exists(template_file):
        print(f"Found existing {template_file}, skipping API call.")
        with open(template_file, "r") as f:
            return [line.strip() for line in f.readlines()]

    print("Calling Gemini API for synthetic templates...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    response = model.generate_content(GEMINI_PROMPT)
    templates = [line.strip() for line in response.text.split('\n') 
                 if line.strip() and not line.startswith('*')]

    # Save templates
    os.makedirs(os.path.dirname(template_file), exist_ok=True)
    with open(template_file, "w") as f:
        for t in templates:
            f.write(t + "\n")

    return templates


def gen_machine_strings():
    """Generate high-entropy machine data strings (URLs, error codes)."""
    generators = [
        lambda: f"Error_0x{random.randint(0,255):02X}_Timeout",
        lambda: f"{random.choice(['http', 'https'])}://api.sensor-cloud.org/v1/dev/{random.randint(1000,9999)}"
    ]
    return random.choice(generators)()


def hydrate_template(template):
    """Replace placeholders in template with random values."""
    vocab = {
        "<SENSOR_TYPE>": ["Lidar", "Thermistor", "Piezo", "Gyroscope", "Barometer"],
        "<OBJECT>": ["Lidar", "Thermistor", "Piezo", "Gyroscope", "Barometer"],
        "<PROPERTY>": ["displacement", "voltage", "amperage", "frequency", "pressure"],
        "<LOCATION>": ["Sector-7", "Edge-Node-B", "Roof-South", "Server-Rack-4"],
        "<UNIT>": ["mV", "mA", "kPa", "Hz", "degC"],
        "<VALUE>": ["HIGH", "LOW", "NORMAL", "CRITICAL"]
    }

    # Replace semantic tags
    for key, words in vocab.items():
        while key in template:
            template = template.replace(key, random.choice(words), 1)

    # Replace ID tag with random numbers
    if "<ID>" in template:
        template = template.replace("<ID>", f"#{random.randint(100, 99999)}")

    return template


def generate_dataset(num_lines, template_set, hybrid_ratio=0.4):
    """
    Generate a hybrid dataset of machine data and natural language.
    
    Args:
        num_lines: Number of lines to generate
        template_set: List of templates to use
        hybrid_ratio: Ratio of machine data (default 0.4 = 40%)
    """
    dataset = []
    for _ in range(num_lines):
        r = random.random()
        if r < hybrid_ratio:
            dataset.append(gen_machine_strings())
        else:
            tmpl = random.choice(template_set)
            dataset.append(hydrate_template(tmpl))
    return dataset


def prepare_datasets(
    template_file, 
    train_path, 
    val_path, 
    test_path,
    train_split=0.8,
    val_split=0.1,
    train_lines=50000,
    val_lines=2000,
    test_lines=2000,
    hybrid_ratio=0.4
):
    """
    Load templates and generate train/val/test datasets.
    
    Args:
        template_file: Path to templates file
        train_path: Output path for training data
        val_path: Output path for validation data
        test_path: Output path for test data
        train_split: Fraction of templates for training (default 0.8)
        val_split: Fraction of templates for validation (default 0.1)
        train_lines: Number of training examples to generate (default 50000)
        val_lines: Number of validation examples to generate (default 2000)
        test_lines: Number of test examples to generate (default 2000)
        hybrid_ratio: Ratio of machine data (default 0.4 = 40%)
    """
    if not os.path.exists(template_file):
        print(f"Error: {template_file} not found. Run Gemini generation first.")
        return

    with open(template_file, "r", encoding="utf-8") as f:
        all_templates = [line.strip() for line in f if line.strip()]

    random.shuffle(all_templates)

    # Split templates
    train_split_idx = int(len(all_templates) * train_split)
    val_split_idx = int(len(all_templates) * (train_split + val_split))

    train_templates = all_templates[:train_split_idx]
    val_templates = all_templates[train_split_idx:val_split_idx]
    test_templates = all_templates[val_split_idx:]

    print(f"Total Templates: {len(all_templates)}")
    print(f"Training Templates: {len(train_templates)}")
    print(f"Validation Templates: {len(val_templates)}")
    print(f"Testing Templates: {len(test_templates)}")

    # Generate datasets
    print(f"Generating Training Data ({train_lines:,} lines)...")
    train_lines_data = generate_dataset(train_lines, train_templates, hybrid_ratio)

    print(f"Generating Validation Data ({val_lines:,} lines)...")
    val_lines_data = generate_dataset(val_lines, val_templates, hybrid_ratio)

    print(f"Generating Testing Data ({test_lines:,} lines)...")
    test_lines_data = generate_dataset(test_lines, test_templates, hybrid_ratio)

    # Save datasets
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines_data))

    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines_data))

    with open(test_path, "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines_data))

    print(f"Saved {train_path}, {val_path}, {test_path}")
