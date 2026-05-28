# %%
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- 1. 설정: 확인하고 싶은 변조 방식 선택 ---
TARGET_MOD = 'QAM16' # '8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 
#'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM' 중에서 선택

MIN_SNR = 10         # 이 숫자 이상의 SNR만 추출합니다.

FILENAME = 'RML2016.10a_dict.dat'

# --- 2. 데이터 로드 및 분리 (학습 코드와 동일 조건) ---
print(f"📂 데이터 로딩 중... (SNR {MIN_SNR}dB 이상의 깨끗한 신호만 필터링합니다)")
try:
    with open(FILENAME, 'rb') as f:
        Xd = pickle.load(f, encoding='latin1')
except FileNotFoundError:
    print(f"❌ '{FILENAME}' 파일이 없습니다.")
    exit()

# 데이터 구조 정리
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):
            lbl.append((mod, snr))
X = np.vstack(X)

# 학습 때와 동일한 시드(2016)로 분리하여 '안 배운 데이터' 확보
X_train, X_test, lbl_train, lbl_test = train_test_split(
    X, lbl, test_size=0.2, random_state=2016
)

# --- 3. 고품질 신호(High SNR) 인덱스 필터링 ---
# 변조 방식이 일치하면서, SNR이 MIN_SNR 이상인 데이터만 수집
high_snr_indices = [
    i for i, label in enumerate(lbl_test) 
    if label[0] == TARGET_MOD and label[1] >= MIN_SNR
]

if len(high_snr_indices) == 0:
    print(f"❌ 조건(Mod: {TARGET_MOD}, SNR >= {MIN_SNR})에 맞는 테스트 데이터가 없습니다.")
    exit()

# 필터링된 고품질 데이터 중 하나 랜덤 선택
idx = np.random.choice(high_snr_indices)
sample_data = X_test[idx]  # (2, 128)
true_label = lbl_test[idx] # (Mod, SNR)

# --- 4. AI 예측 ---
print(f"🧠 AI 분석 중... (정답: {true_label[0]}, 실제 SNR: {true_label[1]}dB)")

# 형태 변환 (2, 128) -> (1, 128, 2)
ai_input = sample_data.T 
input_tensor = ai_input[np.newaxis, ...] 

try:
    model = tf.keras.models.load_model('my_amc_model.h5')
except:
    print("❌ 모델 파일('my_amc_model.h5')을 찾을 수 없습니다.")
    exit()

prediction = model.predict(input_tensor, verbose=0)
pred_idx = np.argmax(prediction)
pred_label = mods[pred_idx]
confidence = np.max(prediction) * 100


# --- 5. 시각화 및 결과 저장 ---
plt.figure(figsize=(12, 6))
is_correct = (true_label[0] == pred_label)
color = 'blue' if is_correct else 'red'

plt.suptitle(f"High SNR Test: {true_label[0]} ({true_label[1]}dB) \n AI Prediction: {pred_label} ({confidence:.1f}%)", 
             fontsize=14, fontweight='bold', color=color)

# 타임 도메인 파형
plt.subplot(1, 2, 1)
plt.plot(ai_input[:, 0], label='I', color='blue', alpha=0.7)
plt.plot(ai_input[:, 1], label='Q', color='red', alpha=0.7, linestyle='--')
plt.title("Waveform (Time Domain)")
plt.grid(True, alpha=0.3)
plt.legend()

# 성상도
plt.subplot(1, 2, 2)
plt.scatter(ai_input[:, 0], ai_input[:, 1], c=range(128), cmap='viridis', edgecolors='k')
plt.axhline(0, color='gray', linewidth=1); plt.axvline(0, color='gray', linewidth=1)
plt.title("Constellation (I vs Q)")
plt.axis('equal')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('high_snr_test_result.png')
plt.show()
print(f"\n📸 결과가 'high_snr_test_result.png'로 저장되었습니다!")
print(f"결과: {true_label[0]} ({true_label[1]}dB) -> 예측: {pred_label} [{'CORRECT' if is_correct else 'WRONG'}]")

# %%
