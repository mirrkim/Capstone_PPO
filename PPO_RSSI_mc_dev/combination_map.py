import numpy as np
import scipy.ndimage
import os
import tifffile
import osmnx as ox
from shapely.geometry import Point
import requests
import time

#======== 공통 설정 값 =======

TAREGET_SIZE = 700
SAVE_DIR ="data"
FALLBACK_API_KEY = "19c8ec4f172050fed3736d53c0efd6c0"

#======= 북한산 위도 및 경도, 고도 조절 =============

TARGET_LAT1 = 37.6587 #위도
TARGET_LON1 = 126.9780 # 경도

# 💡 고도 제한을 450m로 올려서 길이 뚫리게 설정
ALTITUDE_LIMIT = 500   # 고도 조절
MICRO_DENSITY  = 0.01  # 임시 장애물 밀도 조절
LOWLAND_PERCENTILE = 10  #하위 10% 구역을 저지대

#========== 광운대 위도/경도 (osmnx) ==========
TARGET_LAT2 = 37.6194  # 광운대 중심 위도
TARGET_LON2 = 127.0598 # 광운대 중심 경도
RADIUS_M = 200       # 반경 300m 추출
GRID_SIZE = 700      # 100x100 그리드로 쪼개기 (해상도)


def crop_quadrant(terrain, quadrant="SW"):
    h, w = terrain.shape
    quadrant_map = {
        "NW": terrain[:h//2, :w//2], "NE": terrain[:h//2, w//2:],
        "SW": terrain[h//2:, :w//2], "SE": terrain[h//2:, w//2:],
    }
    return quadrant_map.get(quadrant, quadrant_map["SW"])

def fetch_real_srtm_data(lat, lon, size_km=3):
    api_key = os.environ.get("SRTM_API_KEY", FALLBACK_API_KEY).strip()
    if not api_key: return None
    lat_delta, lon_delta = (size_km / 2) / 111, (size_km / 2) / 88
    url = f"https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1&south={lat-lat_delta}&north={lat+lat_delta}&west={lon-lon_delta}&east={lon+lon_delta}&outputFormat=GTiff&API_Key={api_key}"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open("temp.tif", "wb") as f: f.write(response.content)
            terrain = tifffile.imread("temp.tif")
            os.remove("temp.tif")
            return terrain.astype(np.float32)
    except Exception as e: 
        print(f"⚠️ 오류: {e}")
    return None

def build_hybrid_map(terrain, altitude_limit, micro_density):
    lowland_threshold = np.percentile(terrain, LOWLAND_PERCENTILE)
    lowland_map = (terrain < lowland_threshold).astype(int)

    macro_map = (terrain > altitude_limit).astype(int)
    rand_matrix = np.random.rand(*terrain.shape)
    micro_map = ((rand_matrix < micro_density) & (macro_map == 0) & (lowland_map == 0)).astype(int)

    hybrid_map = np.zeros_like(terrain, dtype=int)
    hybrid_map[macro_map == 1] = 1
    hybrid_map[micro_map == 1] = 1
    hybrid_map[lowland_map == 2] = 2  # 저지대 구역은 2로 설정

    print(f"  └ 🟫 일반 지면(0): {(hybrid_map == 0).sum()}칸")
    print(f"  └ 🚧 장애물(1): {(hybrid_map == 1).sum()}칸")
    print(f"  └ 🟨 저지대(2): {(hybrid_map == 2).sum()}칸")
    return hybrid_map


def _resize_map(terrain_map, target_size = TAREGET_SIZE):
    current_h, current_w = terrain_map.shape
    zoom_factor_h = target_size / current_h
    zoom_factor_w = target_size / current_w
    resized_map = scipy.ndimage.zoom(terrain_map, (zoom_factor_h, zoom_factor_w), order=0)
    return resized_map

def generate_enivronment_map(map_type):
    result_map = None

    if map_type in ['SW','SE','NW','NE']:
        terrain = fetch_real_srtm_data(TARGET_LAT1,TARGET_LON1)
        croopped_terrain = crop_quadrant(terrain, map_type)
        raw_hybrid_map = build_hybrid_map(croopped_terrain, ALTITUDE_LIMIT, MICRO_DENSITY)
        result_map = _resize_map(raw_hybrid_map,TAREGET_SIZE)

    elif map_type == 'KWU':
        tags = {'building': True}
        buildings = ox.features_from_point((TARGET_LAT2, TARGET_LON2), tags=tags, dist=RADIUS_M)
        unified_buildings = buildings.geometry.unary_union
        west, south, east, north = buildings.total_bounds

        x_coords = np.linspace(west, east, GRID_SIZE)
        y_coords = np.linspace(north, south, GRID_SIZE) 

        urban_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        for y_idx, y in enumerate(y_coords):
            for x_idx, x in enumerate(x_coords):
                point = Point(x, y)
                if unified_buildings.intersects(point):
                    urban_map[y_idx, x_idx] = 1  
        result_map = urban_map
    
    save_path = os.path.join(SAVE_DIR, f"{map_type}_map.npy")
    np.save(save_path, result_map)
    print(f"  ✅ 저장 완료: {save_path}")


# ==========================================
# 🚀 스크립트 단독 실행부: 5개 맵 모두 생성 후 저장
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("🗺️ 캡스톤 맵 데이터 일괄 생성 시작 (Pre-computing)")
    print("==================================================")
    
    # 폴더가 없으면 자동 생성
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    start_time = time.time()
    map_list = ['SW', 'SE', 'NW', 'NE', 'KWU']
    
    # 리스트에 있는 5개 맵을 차례대로 생성하고 파일로 구워냅니다.
    for map_type in map_list:
        print(f"\n▶ [{map_type}] 지형 데이터 생성 중...")
        generate_enivronment_map(map_type)
        
    end_time = time.time()
    print("\n==================================================")
    print(f"🎉 5개 맵 모두 생성 및 저장 완료! (총 소요 시간: {end_time - start_time:.2f}초)")
    print(f"📂 파일이 저장된 폴더: '{SAVE_DIR}'")
    print("==================================================")