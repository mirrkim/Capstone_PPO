import numpy as np
import matplotlib.pyplot as plt

# 정확도 데이터 로드
accuracy_data = np.load('accuracy_history.npy')

plt.figure(figsize=(10, 5))
plt.plot(accuracy_data, color='dodgerblue', linewidth=2, label='Success Rate')
plt.title('Training Accuracy History (PPO Drone)', fontsize=14, fontweight='bold')
plt.xlabel('Training Updates', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(-5, 105) # 0% ~ 100% 가독성 확보
plt.legend()
plt.show()