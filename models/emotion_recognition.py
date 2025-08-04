# models/emotion_recognition.py
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 7)  # 7 emotions

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EmotionRecognitionModel:
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = EmotionCNN().to(device)
        
        # Load pre-trained weights if available
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded emotion model from {model_path}")
            except Exception as e:
                print(f"Error loading emotion model: {e}")
                print("Using untrained model - results will be random!")
        else:
            print("No emotion model path provided - using untrained model!")
            
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def predict(self, face_img):
        """
        Predict emotions from a face image
        
        Args:
            face_img: Grayscale face image (48x48)
            
        Returns:
            Tensor with emotion probabilities
        """
        # Ensure the image is 48x48
        if face_img.shape != (48, 48):
            face_img = cv2.resize(face_img, (48, 48))
            
        # Normalize pixel values to [0, 1]
        if face_img.max() > 1.0:
            face_img = face_img / 255.0
            
        # Convert to tensor
        face_tensor = torch.tensor(face_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(face_tensor)
            
        # Apply softmax to get probabilities
        probs = F.softmax(output, dim=1)[0]
        return probs