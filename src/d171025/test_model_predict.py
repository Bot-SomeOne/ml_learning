import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import os

def test_model(threshold=0.8):
    # Thiết lập device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f'Using device: {device}')
    # Load checkpoint
    checkpoint = torch.load('catdog_resnet18_best.pt', map_location=device)    
    # Load mô hình
    model = models.resnet18(weights=None)
    num_classes = len(checkpoint['classes'])
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Load weights từ checkpoint
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()    
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['acc']:.4f}")
    print(f"Classes: {checkpoint['classes']}")
    print(f"Using confidence threshold: {threshold}")
    # Chuẩn bị dữ liệu test
    IMG_SIZE = checkpoint.get('img_size', 224)
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder('data/test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # Khởi tạo biến đếm
    correct_confident = 0        # số dự đoán đúng và confident
    total = 0                    # tổng số ảnh
    total_confident = 0          # tổng số dự đoán confident
    uncertain_count = 0          # tổng số dự đoán uncertain
    class_correct = {i: 0 for i in range(num_classes)}   # đúng & confident theo class
    class_total = {i: 0 for i in range(num_classes)}     # tổng ảnh theo class
    class_wrong = {i: 0 for i in range(num_classes)}     # sai & confident theo class
    class_uncertain = {i: 0 for i in range(num_classes)} # uncertain theo class
    # Test mô hình với threshold trên softmax probability
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probs, 1)
            
            total += labels.size(0)
            confident_mask = max_probs >= threshold
            total_confident += confident_mask.sum().item()
            uncertain_count += (~confident_mask).sum().item()
            
            # Đếm cho từng class
            for label, pred, conf in zip(labels, predicted, confident_mask):
                label_item = label.item()
                class_total[label_item] += 1
                if conf.item():  # confident
                    if pred.item() == label_item:
                        class_correct[label_item] += 1
                        correct_confident += 1
                    else:
                        class_wrong[label_item] += 1
                else:
                    class_uncertain[label_item] += 1
    
    # Tính toán accuracy
    accuracy_including_uncertain_as_wrong = 100 * correct_confident / total if total > 0 else 0.0
    accuracy_confident_only = 100 * correct_confident / total_confident if total_confident > 0 else 0.0
    
    # In kết quả
    print('\n' + '='*50)
    print('KẾT QUẢ TEST MÔ HÌNH (VỚI CONFIDENCE THRESHOLD)')
    print('='*50)
    print(f'\nTổng số ảnh: {total}')
    print(f'Threshold: {threshold}')
    print(f'Tổng dự đoán confident: {total_confident}')
    print(f'Tổng dự đoán uncertain: {uncertain_count}')
    print(f'Accuracy (treat uncertain as WRONG): {accuracy_including_uncertain_as_wrong:.2f}% ({correct_confident}/{total})')
    if total_confident > 0:
        print(f'Accuracy (confident-only, exclude uncertain): {accuracy_confident_only:.2f}% ({correct_confident}/{total_confident})')
    else:
        print('Không có dự đoán nào đạt ngưỡng confident; không thể tính accuracy confident-only.')
    
    print('\n' + '-'*50)
    print('CHI TIẾT THEO TỪNG CLASS:')
    print('-'*50)
    
    class_names = test_dataset.classes
    for idx, class_name in enumerate(class_names):
        tot = class_total[idx]
        corr = class_correct[idx]
        wrong = class_wrong[idx]
        un = class_uncertain[idx]
        class_acc_confident = 100 * corr / (corr + wrong) if (corr + wrong) > 0 else 0.0
        print(f'\nClass: {class_name.upper()}')
        print(f'  Tổng số ảnh: {tot}')
        print(f'  Đoán đúng (confident): {corr} ảnh')
        print(f'  Đoán sai (confident): {wrong} ảnh')
        print(f'  Uncertain (prob < {threshold}): {un} ảnh')
        if (corr + wrong) > 0:
            print(f'  Accuracy trên các dự đoán confident của lớp: {class_acc_confident:.2f}% ({corr}/{corr+wrong})')
        else:
            print('  Không có dự đoán confident cho lớp này.')
    
    print('\n' + '='*50)

if __name__ == '__main__':
    THRESHOLD = 0.6
    test_model(threshold=THRESHOLD)