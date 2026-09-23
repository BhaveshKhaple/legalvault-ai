"""
Step 2: Preprocess Documents
Convert PDFs to images and generate descriptions with Gemini
"""
import os
import json
from pdf2image import convert_from_path
from google import genai
from google.genai import types

# Initialize the Gemini client (Make sure your GEMINI_API_KEY environment variable is set)
client = genai.Client()

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


def describe_page(image_path):
    """Use Gemini to describe an instruction page."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            "Describe this instruction page."
        ],
        config=types.GenerateContentConfig(
            system_instruction="""Describe this IKEA assembly instruction page in detail.
Include: step numbers, parts shown, tools needed, actions demonstrated, 
quantities (like "2x"), warnings, and part numbers if visible.
This description will help people find this page when they have questions.""",
            max_output_tokens=500
        )
    )
    return response.text


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