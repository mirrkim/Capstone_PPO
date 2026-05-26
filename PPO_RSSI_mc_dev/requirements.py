import subprocess
import sys

def setup_environment():
    python_exe = sys.executable

    print("🧹 RTX 5070(sm_120)을 위한 깨끗한 환경 조성 중...")
    subprocess.run([python_exe, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])

    print("\n📦 필수 라이브러리 설치...")
    subprocess.run([python_exe, "-m", "pip", "install",
        # 기존
        "numpy", "gymnasium", "matplotlib", "scipy", "pygame",
        # combination_map.py 용
        "tifffile",       # GeoTIFF SRTM 파일 읽기
        "osmnx",          # OpenStreetMap 건물 데이터 (광운대 맵)
        "shapely",        # 건물 폴리곤 교차 판정
        "requests",       # OpenTopography API 호출
    
    ])

    print("\n🔥 RTX 5070 전용 PyTorch 설치 중 (CUDA 12.8 기반)...")
    subprocess.run([
        python_exe, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu128"
    ])

    print("\n✅ 모든 라이브러리 세팅이 완료되었습니다! 이제 5070이 불을 뿜을 겁니다.")

if __name__ == "__main__":
    setup_environment()