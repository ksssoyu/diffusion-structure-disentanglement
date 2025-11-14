import matplotlib.pyplot as plt

def increment_step_count(pipe):
    for module in pipe.unet.modules():
        if hasattr(module, "processor") and hasattr(module.processor, "step_count"):
            module.processor.step_count += 1

def reset_step_count(pipe):
    for module in pipe.unet.modules():
        if hasattr(module, "processor") and hasattr(module.processor, "step_count"):
            module.processor.step_count = 0

def visualize_results(image_A, generated_images, ratios):
    plt.figure(figsize=(20, 6))

    # 맨 왼쪽: 원본 A
    plt.subplot(1, len(ratios) + 1, 1)
    plt.imshow(image_A)
    plt.title("Source (A)")
    plt.axis('off')

    # 나머지: 변화 과정
    for i, (r, img) in enumerate(generated_images):
        plt.subplot(1, len(ratios) + 1, i + 2)
        plt.imshow(img)
        label = "Target (B)" if r == 0.0 else f"Inj: {int(r*100)}%"
        plt.title(f"{label}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()