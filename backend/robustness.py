import torch
import torch.nn.functional as F

def fgsm_attack(image_tensor, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image_tensor + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, -2.5, 2.5)

def generate_adversarial_perturbation(model, input_tensor, label_idx=1, epsilon=0.05):
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    logits = model(input_tensor)
    if logits.shape[-1] == 1:
        target = torch.tensor([float(label_idx)], dtype=torch.float, device=input_tensor.device)
        loss = F.binary_cross_entropy_with_logits(logits.view(-1), target)
    else:
        target = torch.tensor([label_idx], dtype=torch.long, device=input_tensor.device)
        loss = F.cross_entropy(logits, target)
        
    model.zero_grad()
    loss.backward()
    data_grad = input_tensor.grad.data
    perturbed = fgsm_attack(input_tensor, epsilon, data_grad)
    return perturbed
