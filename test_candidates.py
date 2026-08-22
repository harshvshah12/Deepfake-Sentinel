import glob
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

device = 'cuda' if torch.cuda.is_available() else 'cpu'

candidates = [
    "prithivMLmods/deepfake-detector-model-v1",
    "Wvolf/ViT_Deepfake_Detection",
    "dima806/deepfake_vs_real_image_detection"
]

real_imgs = glob.glob('ciplab_raw/real_and_fake_face/training_real/*.jpg')[:5]
fake_imgs = glob.glob('ciplab_raw/real_and_fake_face/training_fake/*.jpg')[:5]

for model_id in candidates:
    print(f"\n==========================================")
    print(f"EVALUATING MODEL: {model_id}")
    try:
        proc = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id).to(device)
        model.eval()
        print("id2label:", model.config.id2label)
        
        real_correct = 0
        for p in real_imgs:
            img = Image.open(p).convert('RGB')
            inputs = proc(images=img, return_tensors='pt').to(device)
            with torch.no_grad():
                probs = model(**inputs).logits.softmax(dim=1)[0]
            top_idx = probs.argmax().item()
            label = model.config.id2label[top_idx]
            print(f"  REAL {p.split('/')[-1]} -> {label} (conf: {probs[top_idx]:.3f})")
            if 'real' in label.lower():
                real_correct += 1
                
        fake_correct = 0
        for p in fake_imgs:
            img = Image.open(p).convert('RGB')
            inputs = proc(images=img, return_tensors='pt').to(device)
            with torch.no_grad():
                probs = model(**inputs).logits.softmax(dim=1)[0]
            top_idx = probs.argmax().item()
            label = model.config.id2label[top_idx]
            print(f"  FAKE {p.split('/')[-1]} -> {label} (conf: {probs[top_idx]:.3f})")
            if 'fake' in label.lower() or 'deepfake' in label.lower():
                fake_correct += 1
                
        print(f"Score: Real {real_correct}/5 | Fake {fake_correct}/5")
    except Exception as e:
        print("Error evaluating", model_id, ":", e)
