# Step 1: Install required packages (only once)
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# !pip install ultralytics==8.1.0
# !pip install lapx>=0.5.2
# !unzip "/content/Bone fracture detection.v1i.yolov8.zip" -d /content/

from ultralytics import YOLO

# Step 2: Create data.yaml file
data_yaml_content = """train: /content/train/images
val: /content/valid/images

nc: 1
names: ['fracture']
"""

# Save the YAML file
with open("/content/data.yaml", "w") as f:  #add you path
    f.write(data_yaml_content)

print("✅ data.yaml file created successfully!")

# Step 3: Train YOLOv8 model
def train_model():
    model = YOLO("yolov8l.yaml")  # or 'yolov8n.yaml', 'yolov8s.yaml' for smaller versions
    result = model.train(
        data="/content/data.yaml",
        epochs=500,
        project="bone_fracture_detection",
        name="My_model",
        batch=64,
        device=0,         # GPU (0), CPU use device='cpu'
        patience=300,
        imgsz=350,
        verbose=True,
        val=True
    )
    print("✅ Training complete!")

if __name__ == "__main__":
    train_model()
