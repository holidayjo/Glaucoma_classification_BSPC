import random
import pandas as pd
import cv2
import os
import time
import torch
import numpy as np
from PIL import Image
from utils.dataset import load_backbone
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

def run_multi_model_inference(args, df, raw_model, raw_scaler, num_deep, device):
    """
    Runs an end-to-end inference timing test using all models in the ensemble.
    """
    print("\n=================================================")
    print("⏱️ Running Multi-Model Inference Timing Test...")
    
    sample_row      = df.iloc[0]
    sample_img_name = sample_row["Original Image Name"]
    sample_img_path = os.path.join(args.root_glauc if sample_row["Label"] == 1 else args.root_normal, sample_img_name)
    
    # 1. Load ALL backbones 
    loaded_backbones = []
    for m_name in args.models:
        custom_pth  = args.weights if m_name == "custom_resnet" else None
        net, tfm, _ = load_backbone(m_name, device, custom_pth)
        loaded_backbones.append((net, tfm))

    # 2. Setup the final classifier wrapper
    if args.classifier == "gcfn":
        # Assuming GCFNInferenceWrapper and PassthroughScaler are defined elsewhere in utils.py
        final_model = GCFNInferenceWrapper(raw_model, raw_scaler, num_deep, device) 
        final_scaler = PassthroughScaler()
    else:
        final_model = raw_model
        final_scaler = raw_scaler

    # 3. Start the Inference Timer
    start_time = time.perf_counter()

    raw_img = Image.open(sample_img_path).convert("RGB")
    
    deep_feats_list = []
    with torch.no_grad():
        for net, tfm in loaded_backbones:
            img_tensor = tfm(raw_img).unsqueeze(0).to(device)
            feat = net(img_tensor).view(-1).cpu().numpy()
            deep_feats_list.append(feat)
            
    X_deep_test = np.hstack(deep_feats_list)
    
    # --- NEW LOGIC: Dynamically include/exclude clinical features ---
    if getattr(args, 'exclude_clinical', False):
        X_combined = X_deep_test.reshape(1, -1)
        inference_type = "Image Only"
    else:
        # FIX: Dynamically extract ONLY the requested clinical features
        X_clin_test    = sample_row[args.clinical_feats].values.astype(float) # name of clinical features to include (e.g., CDR, RDR)
        X_combined     = np.hstack([X_deep_test, X_clin_test]).reshape(1, -1)
        inference_type = f"Image + Clinical Fusion ({', '.join(args.clinical_feats)})"

    X_final_scaled = final_scaler.transform(X_combined)
    prob           = final_model.predict_proba(X_final_scaled)[0, 1]

    end_time = time.perf_counter()
    
    print(f"Image: {sample_img_name}")
    print(f"True Label: {sample_row['Label']} | Predicted Prob (Glaucoma): {prob:.4f}")
    print(f"Total Inference Time ({len(args.models)} Model(s), {inference_type}): {(end_time - start_time) * 1000:.2f} ms")
    print("=================================================\n")
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_next_run_dir(base_dir="runs/train", prefix="exp"):
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)]
    nums = [0]
    for d in existing_dirs:
        suffix = d[len(prefix):]
        if suffix.isdigit(): nums.append(int(suffix))
    new_dir = os.path.join(base_dir, f"{prefix}{max(nums) + 1}")
    os.makedirs(new_dir, exist_ok=True)
    return new_dir

def load_dataframe(csv_path):
    df          = pd.read_csv(csv_path)
    df["Label"] = df["Dataset"].map({"glaucoma": 1, "normal": 0})
    df          = df[~df['CDR'].astype(str).str.contains("Error", na=False)]
    df          = df[~df['RDR'].astype(str).str.contains("NaN", na=False)]
    df['CDR']   = pd.to_numeric(df['CDR'], errors='coerce')
    df['RDR']   = pd.to_numeric(df['RDR'], errors='coerce')
    df          = df.dropna(subset=['CDR', 'RDR'])
    print(f"DataFrame loaded from {csv_path}. Valid Shape: {df.shape}")
    return df

class GCFNInferenceWrapper:
    def __init__(self, gcfn_model, scalers, num_deep, device):
        self.model = gcfn_model
        self.scaler_deep, self.scaler_clin = scalers
        self.num_deep = num_deep
        self.device = device

    def predict_proba(self, X_combined):
        Xd = X_combined[:, :self.num_deep]
        Xc = X_combined[:, self.num_deep:]
        Xd_t = torch.tensor(self.scaler_deep.transform(Xd), dtype=torch.float32).to(self.device)
        Xc_t = torch.tensor(self.scaler_clin.transform(Xc), dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(Xd_t, Xc_t)
            probs = torch.softmax(out, dim=1).cpu().numpy()
        return probs

class PassthroughScaler:
    def transform(self, X): return X
    
    
def measure_single_image_inference(image_path, cdr_val, rdr_val, dl_model, tfm, xgb_model, scaler, device):
    """
    Measures the exact inference time for a single image through the entire pipeline.
    """
    print(f"\n--- Measuring Inference Time ---")
    
    # ---------------------------------------------------------
    # WARM-UP (Crucial for accurate GPU timing)
    # ---------------------------------------------------------
    dl_model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        _ = dl_model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # ---------------------------------------------------------
    # 1. Image Preprocessing Time
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    img = Image.open(image_path).convert("RGB")
    input_tensor = tfm(img).unsqueeze(0).to(device)
    if device.type == 'cuda': torch.cuda.synchronize()
    t_preprocess = time.perf_counter() - t0

    # ---------------------------------------------------------
    # 2. Deep Learning Feature Extraction Time
    # ---------------------------------------------------------
    t1 = time.perf_counter()
    with torch.no_grad():
        deep_features = dl_model(input_tensor)
    if device.type == 'cuda': torch.cuda.synchronize()
    t_dl = time.perf_counter() - t1

    deep_features_np = deep_features.view(-1).cpu().numpy()

    # ---------------------------------------------------------
    # 3. XGBoost Classification Time
    # ---------------------------------------------------------
    t2 = time.perf_counter()
    # Combine deep features with CDR and RDR
    tabular_features = np.array([cdr_val, rdr_val])
    final_feature_vector = np.hstack([deep_features_np, tabular_features]).reshape(1, -1)
    
    # Scale and predict
    final_scaled = scaler.transform(final_feature_vector)
    prob = xgb_model.predict_proba(final_scaled)[0, 1]
    t_xgb = time.perf_counter() - t2

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    total_time = t_preprocess + t_dl + t_xgb
    fps = 1.0 / total_time

    print(f"Prediction       : {'Glaucoma' if prob > 0.5 else 'Normal'} (Probability: {prob:.4f})")
    print(f"Preprocessing    : {t_preprocess * 1000:.2f} ms")
    print(f"Deep Learning    : {t_dl * 1000:.2f} ms")
    print(f"XGBoost + Scaling: {t_xgb * 1000:.2f} ms")
    print(f"-----------------------------------")
    print(f"Total Time       : {total_time * 1000:.2f} ms")
    print(f"Estimated FPS    : {fps:.2f} frames per second\n")

    return total_time

def mask_inspection(test_mask_path = "data/2_g1020_origa_refuge/REFUGE/val/Masks_Cropped/V0001.png"):
    mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
    print("Unique pixel values in this mask:", np.unique(mask))


def compare_crop_and_mask(img_path  = "data/2_g1020_origa_refuge/REFUGE/val/Images_Cropped/V0001.jpg",
                          mask_path = "data/2_g1020_origa_refuge/REFUGE/val/Masks_Cropped/V0001.png"):
    # 2. Read and convert the images
    img     = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    mask    = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 3. Create a 1x2 side-by-side grid
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True, 
                        subplot_titles=("Original Retinal Image", "G1020 Mask (Hover for 0, 1, 2)"))

    # 4. Add the Original Image to the left (Column 1)
    fig.add_trace(go.Image(z=img_rgb), row=1, col=1)

    # 5. Add the Mask to the right (Column 2)
    # We use a Heatmap here so we can easily apply the bright 'viridis' color scale to the 0,1,2 values
    fig.add_trace(go.Heatmap(z=mask, colorscale='viridis', showscale=False), row=1, col=2)

    # 6. Formatting: Fix the axes so the images don't look stretched, and flip the Y-axis for the mask
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y', scaleanchor="x", scaleratio=1)

    fig.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10))
    fig.show()
    
    

def measure_inference_time(model, inputs, model_type="pytorch", device=None, num_runs=10):
    """
    Measures the average inference time of a model over multiple runs.
    
    Args:
        model: The trained model (XGBoost or PyTorch).
        inputs: A tuple containing the inputs to pass to the model.
        model_type: "xgboost" or "pytorch". Determines the timing strategy.
        device: The torch.device (required for accurate PyTorch GPU timing).
        num_runs: Number of iterations to average over.
        
    Returns:
        Average inference time in seconds (float).
    """
    fold_time = 0.0

    if model_type == "xgboost":
        # Warmup (optional but good for caching)
        _ = model.predict_proba(*inputs)
        
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model.predict_proba(*inputs)
        fold_time = (time.perf_counter() - start_time) / float(num_runs)
        
    elif model_type == "pytorch":
        # GPU Warmup (Assuming one forward pass was already done prior to calling this)
        if device is not None and device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(*inputs)
                
        # Wait for the GPU to finish all asynchronous tasks
        if device is not None and device.type == 'cuda':
            torch.cuda.synchronize() 
            
        fold_time = (time.perf_counter() - start_time) / float(num_runs)

    return fold_time


def measure_end_to_end_inference(loaded_backbones, final_model, final_scaler, raw_img, clin_feats, device, is_gcfn=False, num_runs=10):
    """
    Measures true end-to-end inference time: Raw Image -> Deep Backbones -> Classifier Fusion -> Prediction.
    
    Args:
        loaded_backbones: List of tuples (net, transform) for the deep learning models.
        final_model: The trained classifier (XGBoost or GCFNInferenceWrapper).
        final_scaler: The scaler used to normalize the combined features.
        raw_img: The raw PIL Image object.
        clin_feats: Numpy array of clinical features (or empty list if excluded).
        device: torch.device ('cuda' or 'cpu').
        is_gcfn: Boolean flag indicating if the final model is PyTorch GCFN.
        num_runs: Number of iterations to average the timing over.
        
    Returns:
        Average total inference time in seconds (float).
    """
    
    # Helper function to execute exactly one full forward pass of the entire pipeline
    def _single_full_pass():
        deep_feats_list = []
        
        # 1. Image Preprocessing & Deep Feature Extraction
        for net, tfm in loaded_backbones: # At here, we are iterating through each backbone of the mode. 
                                          # applying its specific transform, and extracting features.
            img_tensor = tfm(raw_img).unsqueeze(0).to(device)
            feat       = net(img_tensor).view(-1).cpu().numpy()
            deep_feats_list.append(feat)
            
        # 2. Concatenate Deep Features
        X_deep = np.hstack(deep_feats_list)
        
        # 3. Fuse with Clinical Features
        if clin_feats is not None and len(clin_feats) > 0:
            X_combined = np.hstack([X_deep, clin_feats]).reshape(1, -1)
        else:
            X_combined = X_deep.reshape(1, -1)
            
        # 4. Scale and Predict
        X_scaled = final_scaler.transform(X_combined)
        
        if is_gcfn:
            _ = final_model.predict_proba(X_scaled) 
        else:
            _ = final_model.predict_proba(X_scaled)


    # ==========================================
    # WARMUP PHASE
    # PyTorch allocates memory dynamically. We must do a warmup run so 
    # memory allocation doesn't artificially inflate our timing test.
    # ==========================================
    with torch.no_grad():
        _single_full_pass()
        
    if device.type == 'cuda':
        torch.cuda.synchronize() # Wait for GPU to finish the warmup completely

    # ==========================================
    # ACTUAL TIMING LOOP
    # ==========================================
    start_time = time.perf_counter()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _single_full_pass()
            
    if device.type == 'cuda':
        torch.cuda.synchronize() # Wait for GPU to finish all queued async operations

    # Calculate average time
    avg_time_seconds = (time.perf_counter() - start_time) / float(num_runs)
    return avg_time_seconds


def execute_timing_test(args, df, raw_model, raw_scaler, num_deep, device):
    """
    Handles all the complex setup (loading images, initializing vision backbones, 
    extracting clinical features) before running the actual end-to-end timer.
    """
    print("\n=================================================")
    print("⏱️ Running True End-to-End Inference Timing Test...")
    
    sample_row      = df.iloc[0]
    sample_img_name = sample_row["Original Image Name"]
    sample_img_path = os.path.join(args.root_glauc if sample_row["Label"] == 1 else args.root_normal, sample_img_name)
    raw_img         = Image.open(sample_img_path).convert("RGB")

    # Load backbones
    loaded_backbones = []
    for m_name in args.models:
        custom_pth  = args.weights if m_name == "custom_resnet" else None
        net, tfm, _ = load_backbone(m_name, device, custom_pth)
        loaded_backbones.append((net, tfm))

    # Setup wrappers
    if args.classifier == "gcfn":
        final_model  = GCFNInferenceWrapper(raw_model, raw_scaler, num_deep, device) 
        final_scaler = PassthroughScaler()
        is_gcfn      = True
    else:
        final_model  = raw_model
        final_scaler = raw_scaler
        is_gcfn = False

    # Extract specific clinical features dynamically
    if getattr(args, 'exclude_clinical', False):
        clin_feats     = np.array([])
        inference_type = "Image Only"
    else:
        clin_feats     = sample_row[args.clinical_feats].values.astype(float)
        inference_type = f"Image + Clinical Fusion ({', '.join(args.clinical_feats)})"

    # Run the highly-accurate timer
    avg_time_sec = measure_end_to_end_inference(
        loaded_backbones=loaded_backbones, 
        final_model=final_model, 
        final_scaler=final_scaler, 
        raw_img=raw_img, 
        clin_feats=clin_feats, 
        device=device, 
        is_gcfn=is_gcfn, 
        num_runs=10
    )
    
    print(f"Image: {sample_img_name}")
    print(f"True Label: {sample_row['Label']}")
    print(f"True Total Inference Time ({len(args.models)} Deep Model(s), {inference_type}): {avg_time_sec * 1000:.2f} ms per image")
    print("=================================================\n")
    


def log_final_results(results, args, log_path=None):
    """
    Calculates Confidence Intervals from K-Fold results, prints the summary, 
    and appends it to the log file.
    """
    summary_msg      = f"=== Final {args.k_fold}-Fold CV Results (with {args.conf_interval*100}% CI) ===\n {args}\n"
    
    confidence_level = args.conf_interval
    degrees_freedom  = args.k_fold - 1
    
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    
    # Add timing metric if we ran an inference test
    if args.measure_time and args.load_weights_dir is not None:
        metric_keys.append("inference_time_per_img")

    for k in metric_keys:
        vals = [r[k] for r in results]
        mean_val = np.mean(vals)
        std_dev  = np.std(vals, ddof=1)
        
        if k == "inference_time_per_img":
            summary_msg += f"Total Pipeline Inference Time (per image): {mean_val*1000:.2f} ± {std_dev*1000:.2f} ms\n"
        else:
            std_err  = stats.sem(vals) 
            margin_of_error = std_err * stats.t.ppf((1 + confidence_level) / 2., degrees_freedom)
            summary_msg += f"{k.capitalize()}: {mean_val:.4f} ± {std_dev:.4f} (95% CI: {mean_val - margin_of_error:.4f} - {mean_val + margin_of_error:.4f})\n"
            
    print(summary_msg)
    
    if log_path:
        with open(log_path, "a") as f:
            f.write(summary_msg)