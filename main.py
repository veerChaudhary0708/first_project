import torch
from model import StyleTransferModel
from utils import (
    load_and_process_image,
    denormalize_and_save_image,
    content_loss,
    style_loss,
    total_variation_loss
)

def run_style_transfer():
    # ===============================================================
    #  SETTINGS CONTROL PANEL 🎮
    # ===============================================================
    # --- 1. File Paths ---
    STYLE_IMAGE_PATH = "C:\\Users\\veer\\pytorch-neural-style-transfer\\data\\style-images\\wave_crop.jpg"
    CONTENT_IMAGE_PATH = "C:\\Users\\veer\\pytorch-neural-style-transfer\\data\\content-images\\taj_mahal.jpg"
    OUTPUT_IMAGE_PATH = "output.jpg"

    # --- 2. Model Parameters ---
    IMAGE_SIZE = 512
    OPTIMIZER = 'lbfgs' # 'lbfgs' or 'adam'
    STEPS = 150

    # --- 3. Loss Weights (Your secret sauce!) ---
    CONTENT_WEIGHT = 1e2
    STYLE_WEIGHT = 1e10
    TV_WEIGHT = 0.01
    # ===============================================================


    # 1. SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. LOAD IMAGES AND MODEL
    content_image = load_and_process_image(CONTENT_IMAGE_PATH, IMAGE_SIZE).to(device)
    style_image = load_and_process_image(STYLE_IMAGE_PATH, IMAGE_SIZE).to(device)
    model = StyleTransferModel().to(device)

    # 3. INITIALIZE TARGET IMAGE AND OPTIMIZER
    target_image = content_image.clone().requires_grad_(True)
    
    if OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam([target_image], lr=0.1)
    elif OPTIMIZER == 'lbfgs':
        optimizer = torch.optim.LBFGS([target_image], max_iter=STEPS, line_search_fn='strong_wolfe')

    # 4. PRE-CALCULATE ORIGINAL FEATURES
    with torch.no_grad():
        original_content_features, _ = model(content_image)
        _, original_style_features_for_loss = model(style_image)

    print(f"Starting style transfer with {OPTIMIZER}...")

    # 5. THE OPTIMIZATION LOOP
    if OPTIMIZER == 'adam':
        for step in range(1, STEPS + 1):
            generated_content_features, generated_style_features = model(target_image)
            c_loss = content_loss(generated_content_features, original_content_features)
            s_loss = style_loss(generated_style_features, original_style_features_for_loss)
            t_loss = total_variation_loss(target_image)
            total_loss = CONTENT_WEIGHT * c_loss + STYLE_WEIGHT * s_loss + TV_WEIGHT * t_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                print(f"Step {step}/{STEPS}, Total Loss: {total_loss.item():.4f}")
    
    elif OPTIMIZER == 'lbfgs':
        def closure():
            optimizer.zero_grad()
            generated_content_features, generated_style_features = model(target_image)
            c_loss = content_loss(generated_content_features, original_content_features)
            s_loss = style_loss(generated_style_features, original_style_features_for_loss)
            t_loss = total_variation_loss(target_image)
            total_loss = CONTENT_WEIGHT * c_loss + STYLE_WEIGHT * s_loss + TV_WEIGHT * t_loss
            total_loss.backward()
            print(f"L-BFGS | Total Loss: {total_loss.item():.4f}")
            return total_loss
        optimizer.step(closure)

    print("Optimization finished.")

    # 6. SAVE THE FINAL IMAGE
    denormalize_and_save_image(target_image, OUTPUT_IMAGE_PATH)
    print(f"Image saved to {OUTPUT_IMAGE_PATH}")


if __name__ == '__main__':
    run_style_transfer()