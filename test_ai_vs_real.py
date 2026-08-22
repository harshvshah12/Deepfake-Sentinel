import glob
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = "dima806/ai_vs_real_image_detection"

proc = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id).to(device)
model.eval()

print("Model id2label:", model.config.id2label)

real_imgs = glob.glob('ciplab_raw/real_and_fake_face/training_real/*.jpg')[:6]
fake_imgs = glob.glob('ciplab_raw/real_and_fake_face/training_fake/*.jpg')[:6]

print("\n=== REAL FACES ===")
for p in real_imgs:
    img = Image.open(p).convert('RGB')
    inputs = proc(images=img, return_tensors='pt').to(device)
    with torch.no_grad():
        probs = model(**inputs).logits.softmax(dim=1)[0]
    pred = model.config.id2label[probs.argmax().item()]
    print(f"{p.split('/')[-1]}: Real={probs[0]:.3f}, AI/Fake={probs[1]:.3f} -> {pred}")

print("\n=== FAKE FACES ===")
for p in fake_imgs:
    img = Image.open(p).convert('RGB')
    inputs = proc(images=img, return_tensors='pt').to(device)
    with torch.no_grad():
        probs = model(**inputs).logits.softmax(dim=1)[0]
    pred = model.config.id2label[probs.argmax().item()]
    print(f"{p.split('/')[-1]}: Real={probs[0]:.3f}, AI/Fake={probs[1]:.3f} -> {pred}")
