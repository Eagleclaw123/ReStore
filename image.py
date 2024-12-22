import cv2
import numpy as np
import os
import time
import torch
from PIL import Image
from torchvision import transforms
import pytesseract
import easyocr  # Advanced OCR library
from pdf2image import convert_from_path
from multiprocessing import Pool

# Configure Tesseract and EasyOCR
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
POPPLER_PATH = r"C:\\Program Files\\poppler-24.08.0\\poppler-24.08.0\\Library\\bin"
reader = easyocr.Reader(['en'], gpu=False)  # Use GPU for EasyOCR if available

# LaMa Model Path
LAMA_MODEL_PATH = r"C:\\Users\\91996\\.cache\\torch\\hub\\checkpoints\\big-lama.pt"

# Load LaMa Model
def load_lama_model(model_path):
    """Load the LaMa model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model, device

# Inpainting with LaMa while preserving resolution
def inpaint_with_lama(image, mask, model, device):
    """Restore the design using the LaMa model while preserving resolution."""
    # Original image size
    original_size = image.size  # (width, height)

    # Transform input to tensor resized for model inference
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),  # Resize for model compatibility
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    mask_tensor = transform(mask).unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        inpainted_tensor = model(image_tensor, mask_tensor)

    # Convert tensor back to image
    inpainted_image = inpainted_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    inpainted_image = (inpainted_image * 255).astype(np.uint8)

    # Resize back to original resolution
    inpainted_image_resized = cv2.resize(inpainted_image, original_size, interpolation=cv2.INTER_LINEAR)
    return inpainted_image_resized

# Load LaMa model
lama_model, lama_device = load_lama_model(LAMA_MODEL_PATH)

def extract_images_from_pdf(pdf_path, dpi=200):
    """Extract images from a PDF file."""
    print("Extracting images from PDF...")
    images = convert_from_path(pdf_path, dpi=dpi, poppler_path=POPPLER_PATH)
    return images

def detect_text_regions_easyocr(image):
    """Detect text regions using EasyOCR."""
    results = reader.readtext(image, detail=1, paragraph=False)
    regions = []
    for (bbox, text, prob) in results:
        if prob > 0.5:  # Filter out low-confidence detections
            (top_left, top_right, bottom_right, bottom_left) = bbox
            x1, y1 = map(int, top_left)
            x2, y2 = map(int, bottom_right)
            regions.append((x1, y1, x2, y2))
    return regions

def create_fine_mask(image, text_regions):
    """Create a refined mask for text regions."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for (x1, y1, x2, y2) in text_regions:
        cv2.rectangle(mask, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), 255, thickness=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=2)  # Adjust for better smoothing
    return mask

def process_image(image_cv):
    """Full pipeline to remove text and restore the design."""
    text_regions = detect_text_regions_easyocr(image_cv)
    if not text_regions:
        return image_cv  # Skip processing if no text is detected
    
    # Create a mask
    mask = create_fine_mask(image_cv, text_regions)
    
    # Convert OpenCV image and mask to PIL format
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask)

    # Inpaint using LaMa
    inpainted = inpaint_with_lama(image_pil, mask_pil, lama_model, lama_device)
    return cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR)

def process_images_parallel(images):
    """Process images in parallel."""
    with Pool(processes=2) as pool:  # Limit to 2 processes to reduce CPU usage
        return pool.map(process_image, images)

def save_cleaned_images_to_folder(images, output_folder):
    """Save each cleaned image to a folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for idx, img in enumerate(images):
        output_path = os.path.join(output_folder, f"cleaned_image_{idx + 1}.png")
        cv2.imwrite(output_path, img)
        print(f"Saved cleaned image: {output_path}")

def main(input_pdf, output_folder):
    """Main workflow."""
    start_time = time.time()
    images = extract_images_from_pdf(input_pdf, dpi=200)
    print(f"{len(images)} images extracted. Starting processing...")

    # Convert PIL images to OpenCV format for processing
    images_cv = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]

    # Process images in parallel
    cleaned_images = process_images_parallel(images_cv)

    # Save cleaned images to the output folder
    save_cleaned_images_to_folder(cleaned_images, output_folder)
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")
    print("All images processed and outpainted successfully!")

# Replace with your actual paths
if __name__ == "__main__":
    input_pdf = r"C:\\Users\\91996\\Desktop\\images1.pdf"
    output_folder = r"C:\\Users\\91996\\Desktop\\imagestest"
    main(input_pdf, output_folder)
