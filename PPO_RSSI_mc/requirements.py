import subprocess
import sys

def setup_environment():
    # 현재 실행 중인 파이썬 환경의 pip를 사용하도록 sys.executable 사용
    python_exe = sys.executable

    print("🧹 기존 PyTorch 관련 라이브러리 삭제 중...")
    subprocess.run([python_exe, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])

    print("\n📦 필수 라이브러리 설치 중 (numpy, gymnasium)...")
    subprocess.run([python_exe, "-m", "pip", "install", "numpy", "gymnasium"])

    print("\n🔥 CUDA 버전에 맞는 PyTorch 설치 중 (기본: CUDA 12.1)...")
    print("\n🔥 설치가 좀 걸림")
    # 만약 본인 컴퓨터의 CUDA 버전이 11.8이라면, 아래 코드 맨 끝의 'cu121'을 'cu118'로 수정해줘!
    subprocess.run([
        python_exe, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ])

    print("\n✅ 모든 라이브러리 세팅이 완료되었어!")

if __name__ == "__main__":
    setup_environment()