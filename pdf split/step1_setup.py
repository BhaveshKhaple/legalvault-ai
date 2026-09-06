"""
Step 1: Setup and Download Sample Data
"""
import os
import requests

# Create directories
for d in ["data", "data/images", "data/cache"]:
    os.makedirs(d, exist_ok=True)
print("✓ Created directories")

# Download IKEA assembly instructions
SAMPLE_PDFS = {
    "malm_desk.pdf": "https://www.ikea.com/us/en/assembly_instructions/malm-desk-white__AA-516949-7-2.pdf",
    "malm_bed.pdf": "https://www.ikea.com/us/en/assembly_instructions/malm-bed-frame-low__AA-75286-15_pub.pdf",
}

for filename, url in SAMPLE_PDFS.items():
    filepath = f"data/{filename}"
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        response = requests.get(url, timeout=30)
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"  ✓ Saved ({len(response.content) / 1024:.1f} KB)")
    else:
        print(f"✓ {filename} already exists")