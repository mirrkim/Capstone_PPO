import subprocess
import sys

def install_requirements():
    packages = [
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "matplotlib>=3.5.0"
    ]

    print("=== 🚀 드론 시뮬레이션 필수 라이브러리 자동 설치를 시작합니다 ===")
    for package in packages:
        print(f"\n📦 설치 및 확인 중: {package} ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ [{package}] 설치 완료 (또는 이미 최신 버전입니다)!")
        except subprocess.CalledProcessError:
            print(f"❌ [{package}] 설치 실패. 에러 메시지를 확인해주세요.")

    print("\n🎉 모든 준비가 끝났습니다! train.py를 실행하세요.")

if __name__ == "__main__":
    install_requirements()
