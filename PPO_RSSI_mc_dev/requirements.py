import subprocess
import sys

def setup_environment():
    python_exe = sys.executable

    print("🧹 RTX 5070(sm_120)을 위한 깨끗한 환경 조성 중...")
    subprocess.run([python_exe, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])

    print("\n📦 필수 라이브러리 설치 (numpy, gymnasium, matplotlib, scipy)...")
    # 시각화(matplotlib)와 데이터 로드(scipy)를 위해 추가했습니다.
    subprocess.run([python_exe, "-m", "pip", "install", "numpy", "gymnasium", "matplotlib", "scipy"])

    print("\n🔥 RTX 5070 전용 PyTorch 설치 중 (CUDA 12.8 기반)...")
    # cu121 대신 5070을 지원하는 cu128 인덱스를 사용합니다.
    subprocess.run([
        python_exe, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu128" 
    ])

    print("\n✅ 모든 라이브러리 세팅이 완료되었습니다! 이제 5070이 불을 뿜을 겁니다.")

if __name__ == "__main__":
    setup_environment()