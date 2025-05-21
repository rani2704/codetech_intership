import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape:
        size = shape  # must be tuple (height, width)

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    image = transform(image)[:3, :, :].unsqueeze(0)
    return image.to(device)

def im_convert(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image * torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    image = image + torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    image = image.clamp(0, 1)
    return transforms.ToPILImage()(image)

# Load images
content = load_image("tajmahal.jpg").to(device)
style = load_image("vangogh.jpg", shape=(content.size(2), content.size(3))).to(device)

# Load VGG19 model
vgg = models.vgg19(weights='VGG19_Weights.DEFAULT').features.to(device).eval()

# Freeze parameters
for param in vgg.parameters():
    param.requires_grad = False

# Content and style layers
content_layers = ['conv4_2']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# Get features
def get_features(image, model, layers):
    features = {}
    x = image
    i = 0
    for layer in model.children():
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv{i}_1'
            if name in layers:
                features[name] = x
    return features

# Gram matrix
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())

# Extract features
content_features = get_features(content, vgg, content_layers)
style_features = get_features(style, vgg, style_layers)

# Compute gram matrices for style features
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Target image
target = content.clone().requires_grad_(True).to(device)

# Optimizer and loss
optimizer = optim.Adam([target], lr=0.003)
style_weight = 1e6
content_weight = 1

# Optimization loop
for step in range(1, 201):
    target_features = get_features(target, vgg, content_layers + style_layers)

    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

    style_loss = 0
    for layer in style_layers:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_style_loss / (target_feature.shape[1] ** 2)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Total Loss: {total_loss.item()}")

# Save output
output = im_convert(target)
output.save("stylized_output.jpg")
print("Stylized image saved as stylized_output.jpg")
