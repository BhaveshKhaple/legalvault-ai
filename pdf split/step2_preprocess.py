"""
Step 2: Preprocess Documents
Convert PDFs to images and generate descriptions with Gemini
"""
import os
import json
from pdf2image import convert_from_path
import ollama


DATA_DIR = "data"
IMAGES_DIR = "data/images"
CACHE_FILE = "data/cache/descriptions.json"


def convert_pdf_to_images(pdf_path):
    """Convert a PDF to page images."""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    images = convert_from_path(pdf_path, dpi=150)
    
    image_paths = []
    for i, image in enumerate(images):
        image_path = f"{IMAGES_DIR}/{pdf_name}_page_{i+1:03d}.png"
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    
    return image_paths


import time

def describe_page(image_path):
    """Use Local Ollama VLM to describe an instruction page."""
    
    # We pass the file path directly; the Ollama Python client handles the rest
    response = ollama.chat(
        model="gemma4:e2b-it-qat", 
        messages=[
            {
                "role": "system",
                "content": """Describe this IKEA assembly instruction page in detail.
Include: step numbers, parts shown, tools needed, actions demonstrated, 
quantities (like "2x"), warnings, and part numbers if visible."""
            },
            {
                "role": "user",
                "content": "Describe this instruction page.",
                "images": [image_path] # Attach the image path here
            }
        ]
    )
    
    return response['message']['content']


def main():
    # Load cache if exists
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
    
    all_pages = []
    
    # Process each PDF
    for pdf_file in os.listdir(DATA_DIR):
        if not pdf_file.endswith(".pdf"):
            continue
        
        print(f"\nProcessing {pdf_file}...")
        pdf_path = f"{DATA_DIR}/{pdf_file}"
        
        # Convert to images
        print("  Converting to images...")
        image_paths = convert_pdf_to_images(pdf_path)
        print(f"  ✓ {len(image_paths)} pages")
        
        # Generate descriptions
        print("  Generating descriptions with Gemini...")
        for i, image_path in enumerate(image_paths):
            if image_path in cache:
                description = cache[image_path]
                status = "cached"
            else:
                description = describe_page(image_path)
                cache[image_path] = description
                # Save cache after each page
                with open(CACHE_FILE, "w") as f:
                    json.dump(cache, f, indent=2)
                status = "generated"
            
            all_pages.append({
                "source_pdf": pdf_file,
                "page_number": i + 1,
                "image_path": image_path,
                "description": description
            })
            print(f"    Page {i+1}/{len(image_paths)} [{status}]")
    
    # Save all page data
    with open("data/cache/all_pages.json", "w") as f:
        json.dump(all_pages, f, indent=2)
    
    print(f"\n✓ Processed {len(all_pages)} pages total")


if __name__ == "__main__":
    main()