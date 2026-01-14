# industrial-defect-inspector-vit

I utilized the MVTec AD dataset, typically used for unsupervised anomaly detection, and restructured it into a binary classification task. This was done to evaluate the real-time performance of the FastViT architecture in a 'data-scarce' industrial environment where defective samples are limited.


End-to-end industrial anomaly detection system using FastViT, optimized for C++ edge deployment.
The Project Objective
You are building an automated system that looks at a live stream of industrial parts (like bottles or electronics) and identifies if they are "Good" or "Defective" in real-time.

Step-by-Step Execution Plan
Phase 1: The "Intelligence" (Weeks 1–2)
Dataset: You are using MVTec AD. You will treat this as a "Binary Classification" task.

The Model: Instead of a heavy standard Transformer, you will use FastViT (via the timm library).

The Goal: Train the model in Google Colab to achieve high accuracy. You want it to recognize tiny scratches or cracks that shouldn't be there.

Phase 2: The "Software" (Week 2)
The Move: You will download your trained model and move from a Jupyter Notebook to a Python Script (.py).

The Stream: You will use OpenCV to create a "Video Stream." Even if you don't have a real camera, your script will "read" images from a folder one-by-one to simulate a camera on a factory line.

Phase 3: The "Packaging" (Week 3 - MLOps)
Docker: You will wrap your script into a Docker Container.

The Why: This proves to an employer (like Ocado or Dyson) that your code can run on any machine without "it worked on my computer" excuses. This is a core MLOps skill.

Phase 4: The "Speed" (Week 4 - C++ & Optimization)
ONNX: You will convert your PyTorch model into the ONNX format.

C++: You will write a small C++ program to load that ONNX model.

The Flex: This shows you can handle "Edge AI"—running models as fast as possible on hardware where Python might be too slow.
