# 🎭 Deepfake Detection - 빠른 시작 가이드

EfficientNet-B4를 사용한 딥페이크 이미지 탐지 모델

---

## 📦 필요한 라이브러리 설치

```bash
pip install -r requirements.txt
```

## 빠른 테스트
```bash
!git clone https://github.com/HyukjunChoi00/Effi_ViT_DeepFakeImageClassifier
cd Effi_ViT_DeepFakeImageClassifier
pip install -r requirements.txt
python train_deepfake_detection.py
---

## 🚀 빠른 시작 (3단계)

### 1️⃣ 모델 학습

```bash
python train_deepfake_detection.py
```

**예상 시간:** 6-8시간 (GPU)  
**예상 정확도:** 96-98%

### 2️⃣ 학습 결과 확인

학습이 완료되면 `checkpoints_deepfake/` 폴더에 저장됩니다:
```
checkpoints_deepfake/
└── efficientnet_b4_20250111_123456/
    ├── best_model.pth          # 최고 성능 모델
    ├── history.json            # 학습 히스토리
    └── training_curves.png     # 학습 곡선 그래프
```

### 3️⃣ 이미지 예측

```python
import torch
from train_deepfake_detection import predict_image
from efficientnet_from_scratch import efficientnet_b4

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = efficientnet_b4(num_classes=2)

checkpoint = torch.load('checkpoints_deepfake/.../best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

# 이미지 예측
prediction, confidence = predict_image(model, 'test_image.jpg', device)

# 결과 출력
if prediction == 0:
    print(f"🔴 FAKE (신뢰도: {confidence:.2%})")
else:
    print(f"✅ REAL (신뢰도: {confidence:.2%})")
```

---

## 📊 데이터셋 정보

- **출처:** [HuggingFace - JamieWithofs/Deepfake-and-real-images](https://huggingface.co/datasets/JamieWithofs/Deepfake-and-real-images)
- **크기:** 140K 학습 이미지, 39K 검증 이미지, 11K 테스트 이미지
- **Label:**
  - `0` = Fake (딥페이크 이미지)
  - `1` = Real (진짜 이미지)

---

## ⚙️ 설정 변경

`train_deepfake_detection.py` 파일 하단의 `config` 수정:

```python
config = {
    'num_epochs': 20,        # 학습 epoch 수
    'batch_size': 16,        # 배치 크기 (GPU 메모리에 맞게 조정)
    'learning_rate': 1e-4,   # 학습률
    'weight_decay': 1e-5,    # 정규화
    'image_size': 380,       # 입력 이미지 크기 (B4 기본값)
    'save_dir': 'checkpoints_deepfake',  # 저장 경로
}
```


---

## 📈 예상 학습 결과

```
Epoch  1: Train Acc: 65% | Val Acc: 67%
Epoch  5: Train Acc: 82% | Val Acc: 83%
Epoch 10: Train Acc: 92% | Val Acc: 91%
Epoch 15: Train Acc: 96% | Val Acc: 95%
Epoch 20: Train Acc: 97% | Val Acc: 96%

Test Accuracy: 96-98%
예시일 뿐..  
```

---

## 🔍 주요 기능

### 1. 자동 데이터 균형
- Fake와 Real 이미지를 50:50으로 자동 균형
- 모든 split (train/val/test)에 적용

### 2. 데이터 증강
- RandomHorizontalFlip
- RandomRotation (±15°)
- ColorJitter
- RandomAffine

### 3. 학습 최적화
- AdamW Optimizer
- Cosine Annealing Scheduler
- Best Model 자동 저장
- 5 epoch마다 체크포인트 저장

### 4. 상세한 평가
- Confusion Matrix
- Classification Report
- 학습 곡선 그래프

---

## 💡 문제 해결

### CUDA Out of Memory
```python
# batch_size 줄이기
config['batch_size'] = 8  # 또는 4
```

### 데이터셋 다운로드 실패
```bash
# HuggingFace 로그인
huggingface-cli login
```

### 학습 속도가 느림
```python
# num_workers 조정 (CPU 코어 수에 맞게)
# train_deepfake_detection.py Line 472
num_workers=8  # 기본값: 4
```

---

## 📁 필요한 파일

```
your_project/
├── train_deepfake_detection.py  # 학습 스크립트 (필수)
├── efficientnet_from_scratch.py # 모델 구현 (필수)
└── simple_readme.md             # 이 파일
```

---

## 🎯 전체 워크플로우

```
1. 라이브러리 설치
   ↓
2. python train_deepfake_detection.py 실행
   ↓
3. (학습 진행)
   ↓
4. checkpoints_deepfake/ 폴더에서 best_model.pth 확인
   ↓
5. 모델 로드 후 이미지 예측
```

---

## 추가 정보

- **학습 진행 상황 확인:** 터미널에 실시간 progress bar 표시
- **학습 중단 후 재개:** 체크포인트에서 로드 가능
- **다른 모델 크기:** `efficientnet_b0()` ~ `efficientnet_b7()` 사용 가능

---

**🎉 학습 코드**

```bash
python train_deepfake_detection.py
```
