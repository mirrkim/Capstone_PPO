import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- 1. 데이터셋 로드 및 전처리 ---
def load_dataset(filename):
    print("📂 데이터셋을 로드하는 중... (시간이 조금 걸릴 수 있습니다)")
    # RML2016.10a는 Python 2에서 피클링된 경우가 많아 'latin1' 인코딩 필요
    with open(filename, 'rb') as f:
        Xd = pickle.load(f, encoding='latin1')

    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    
    X = []
    lbl = []
    
    # 데이터셋의 모든 (변조방식, SNR) 쌍을 순회하며 데이터 추출
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
    
    X = np.vstack(X)  # 배열 합치기
    
    # 데이터 형태 변환: (N, 2, 128) -> (N, 128, 2)
    # Keras의 Conv1D는 (Time, Channel) 형태를 선호합니다.
    # 논문은 2x128을 이미지처럼 썼지만, 1D Conv로 처리하는 것이 현대적 구현입니다.
    X = np.transpose(X, (0, 2, 1)) 
    
    print(f"✅ 데이터 로드 완료! 형태: {X.shape}")
    return X, lbl, mods, snrs

# --- 2. CNN 모델 설계 (O'Shea 논문 기반) ---
def build_model(input_shape, num_classes):
    # 논문의 구조를 Keras 최신 버전에 맞게 구현 (CNN2 모델) [cite: 162, 170]
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Conv Layer 1: 64 필터 [cite: 177]
        layers.Conv1D(64, 3, padding='same', activation='relu', name="conv1"),
        layers.Dropout(0.5), # 논문에서 강조한 Dropout [cite: 165]
        
        # Conv Layer 2: 16 필터 [cite: 158]
        layers.Conv1D(16, 3, padding='same', activation='relu', name="conv2"),
        layers.Dropout(0.5),
        
        # Flatten & Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu', name="dense1"), # [cite: 54]
        layers.Dropout(0.5),
        
        # Output Layer: Softmax [cite: 163]
        layers.Dense(num_classes, activation='softmax', name="output")
    ])
    
    # 논문에서 사용한 Adam Optimizer [cite: 166]
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# --- 3. 실행 및 시뮬레이션 메인 ---

# [설정] 데이터셋 파일 경로 (같은 폴더에 두세요)
FILENAME = 'RML2016.10a_dict.dat'

try:
    # 1. 데이터 준비
    X, lbl, mods, snrs = load_dataset(FILENAME)
    
    # 레이블 변환 (One-Hot Encoding)
    # 튜플 (Mod, SNR)에서 Mod만 뽑아서 숫자로 변환
    y_indices = np.array(list(map(lambda x: mods.index(x[0]), lbl)))
    y = tf.keras.utils.to_categorical(y_indices, len(mods))
    
    # 학습용/테스트용 분리 (8:2) 
    X_train, X_test, y_train, y_test, lbl_train, lbl_test = train_test_split(
        X, y, lbl, test_size=0.2, random_state=2016
    )
    
    # 2. 모델 학습
    print("\n🧠 CNN 모델 학습 시작...")
    model = build_model((128, 2), len(mods))
    model.summary()
    
    # 논문은 약 70 Epochs 이상 학습했으나, 시뮬레이션 확인용으로 줄임 (성능 원하면 늘리세요)
    history = model.fit(
        X_train, y_train,
        batch_size=1024, # 논문 Batch Size [cite: 172]
        epochs=50,       # 시간 절약을 위해 10번만 (PC 성능 좋으면 50 추천)
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )

    # 3. 분류 시뮬레이션 및 시각화
    print("\n📊 시뮬레이션 결과 분석 중...")
    
    # (1) 혼동 행렬 (Confusion Matrix) - 논문 Figure 8 재현 [cite: 228]
    # 높은 SNR(18dB)인 데이터만 골라서 확인해보기
    test_snrs = map(lambda x: x[1], lbl_test)
    high_snr_idx = [i for i, snr in enumerate(test_snrs) if snr >= 18]
    
    X_test_high = X_test[high_snr_idx]
    y_test_high = np.argmax(y_test[high_snr_idx], axis=1)
    
    y_pred_high = np.argmax(model.predict(X_test_high), axis=1)
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test_high, y_pred_high)
    # 보기 좋게 정규화
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=mods, yticklabels=mods)
    plt.title('Confusion Matrix (SNR >= 18dB)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # (2) 실제 신호 분류 데모
    # 테스트셋에서 랜덤하게 하나 뽑아서 AI가 맞추는지 확인
    demo_idx = np.random.randint(len(X_test))
    demo_signal = X_test[demo_idx]
    true_mod, true_snr = lbl_test[demo_idx]
    
    # AI 예측
    pred_prob = model.predict(demo_signal[np.newaxis, ...])[0]
    pred_mod = mods[np.argmax(pred_prob)]
    confidence = np.max(pred_prob) * 100
    
    # 결과 출력
    plt.figure(figsize=(12, 4))
    plt.plot(demo_signal[:, 0], label='I (In-Phase)')
    plt.plot(demo_signal[:, 1], label='Q (Quadrature)', alpha=0.7)
    plt.title(f"AI Classification Demo\nTrue: {true_mod} ({true_snr}dB) -> Pred: {pred_mod} ({confidence:.1f}%)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"🎯 정답: {true_mod} | 예측: {pred_mod} (확신도: {confidence:.1f}%)")

except FileNotFoundError:
    print(f"❌ 오류: '{FILENAME}' 파일을 찾을 수 없습니다.")
    print("해당 파일을 프로젝트 폴더에 넣고 다시 실행해주세요.")

model.save('my_amc_model.h5')
print("💾 모델이 'my_amc_model.h5' 파일로 저장되었습니다!")