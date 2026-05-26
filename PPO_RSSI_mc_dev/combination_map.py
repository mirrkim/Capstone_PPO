import numpy as np
import scipy.ndimage
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tifffile
import osmnx as ox
from shapely.geometry import Point
import requests

# ======== 동적 경로 설정 (어느 컴퓨터에서든 실행 가능하게) ========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======== 공통 설정 값 =======
TAREGET_SIZE = 700
# 파일이 있는 폴더 안에 'data' 폴더를 생성하도록 절대 경로로 수정
SAVE_DIR = os.path.join(BASE_DIR, "data") 
FALLBACK_API_KEY = "19c8ec4f172050fed3736d53c0efd6c0"

#======= 북한산 위도 및 경도, 고도 조절 =============
TARGET_LAT1 = 37.6587 #위도
TARGET_LON1 = 126.9780 # 경도

# 💡 고도 제한을 450m로 올려서 길이 뚫리게 설정
ALTITUDE_LIMIT = 500   # 고도 조절
MICRO_DENSITY  = 0.01  # 임시 장애물 밀도 조절, 0.03이면 전체 맵에서 3%정도 장애물 생성
# 💡 Water 대신 Lowland(저지대) 명칭 사용
LOWLAND_PERCENTILE = 10  #하위 10% 구역을 저지대 = 드론이 이 고도에서는 움직일 수 있다. 

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
    
    # 임시 저장 파일도 안전한 절대 경로로 지정        
    temp_file_path = os.path.join(BASE_DIR, "temp.tif")

    print("🌐 지형 데이터 요청 중...")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(temp_file_path, "wb") as f: 
                f.write(response.content)
            terrain = tifffile.imread(temp_file_path)
            os.remove(temp_file_path)
            return terrain.astype(np.float32)
        else:
            print(f"⚠️ API 요청 실패: 상태 코드 {response.status_code}")
    except Exception as e: 
        print(f"⚠️ 오류: {e}")
    return None

def build_hybrid_map(terrain, altitude_limit, micro_density):
    # 💡 1. 저지대(Lowland) 맵 생성: 전체 지형 중 하위 10%
    lowland_threshold = np.percentile(terrain, LOWLAND_PERCENTILE)
    lowland_map = (terrain < lowland_threshold).astype(int)

    macro_map = (terrain > altitude_limit).astype(int)
    rand_matrix = np.random.rand(*terrain.shape)
    micro_map = ((rand_matrix < micro_density) & (macro_map == 0) & (lowland_map == 0)).astype(int)

    hybrid_map = np.zeros_like(terrain, dtype=int)
    hybrid_map[macro_map == 1] = 1
    hybrid_map[micro_map == 1] = 1
    hybrid_map[lowland_map == 1] = 2  # 저지대 구역은 2로 설정

    print(f"📊 지형 셀 통계:")
    print(f"  └ 🟫 일반 지면(0): {(hybrid_map == 0).sum()}칸")
    print(f"  └ 🚧 장애물(1): {(hybrid_map == 1).sum()}칸")
    print(f"  └ 🟨 저지대(2): {(hybrid_map == 2).sum()}칸")
    return hybrid_map


#===== 북한산 지형 700*700 으로 사이즈 조절
def _resize_map(terrain_map, target_size = TAREGET_SIZE):
    current_h, current_w = terrain_map.shape
    zoom_factor_h = target_size / current_h
    zoom_factor_w = target_size / current_w

    resized_map = scipy.ndimage.zoom(terrain_map, (zoom_factor_h, zoom_factor_w), order=0)
    return resized_map

def generate_enivronment_map(map_type):
    # map_tpe: 'SW','SE','NW','NE'(북한산), 'KWU'(광운대)
    os.makedirs(SAVE_DIR, exist_ok=True)
    result_map = None

    if map_type in ['SW','SE','NW','NE']:
        print(f"\n⛰️ 북한산 {map_type} 지형 생성 중...")
        terrain = fetch_real_srtm_data(TARGET_LAT1,TARGET_LON1)
        
        # 💡 [핵심 수정] terrain이 None으로 반환되었을 때 에러(shape 불가)를 막는 방어 코드
        if terrain is None:
            print("❌ 지형 데이터를 가져오지 못해 맵 생성을 건너뜁니다.")
            return None

        croopped_terrain = crop_quadrant(terrain, map_type)
        raw_hybrid_map = build_hybrid_map(croopped_terrain, ALTITUDE_LIMIT, MICRO_DENSITY)
        result_map = _resize_map(raw_hybrid_map,TAREGET_SIZE)

    elif map_type == 'KWU':
        print(f"\n🏫 광운대학교 도심지 {map_type} 맵 생성 중 (작은 건물 필터링 적용)...")
        tags = {'building': True}
        buildings = ox.features_from_point((TARGET_LAT2, TARGET_LON2), tags=tags, dist=RADIUS_M)
        
        # ==========================================
        # 💡 수정된 부분: ox.project_gdf -> ox.projection.project_gdf
        # ==========================================
        # 1. 미터 단위로 투영
        buildings_proj = ox.projection.project_gdf(buildings)
        
        # 2. 최소 건물 면적 설정 (예: 100제곱미터)
        MIN_BUILDING_AREA = 100 
        
        # 3. 설정한 면적 이상인 건물만 남기기
        buildings_proj = buildings_proj[buildings_proj.geometry.area >= MIN_BUILDING_AREA]
        
        # 4. 다시 위도/경도 좌표계로 복구
        buildings = ox.projection.project_gdf(buildings_proj, to_latlong=True)
        # ==========================================

        if buildings.empty:
            print("⚠️ 설정한 기준을 만족하는 건물이 없습니다. MIN_BUILDING_AREA를 낮춰보세요.")
            return None

        unified_buildings = buildings.geometry.unary_union
        west, south, east, north = buildings.total_bounds

        x_coords = np.linspace(west, east, GRID_SIZE)
        y_coords = np.linspace(north, south, GRID_SIZE) # 북쪽에서 남쪽으로

        urban_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        
        # 각 칸의 중심점이 건물 폴리곤 안에 들어가는지(장애물인지) 검사
        for y_idx, y in enumerate(y_coords):
            for x_idx, x in enumerate(x_coords):
                point = Point(x, y)
                if unified_buildings.intersects(point):
                    urban_map[y_idx, x_idx] = 1  # 건물이면 1 (장애물)
                    
        result_map = urban_map
    else:
        print("오류: 잘못된 map_type이 입력되었습니다.")
        return None
    
    save_path = os.path.join(SAVE_DIR, f"{map_type}_map.npy")
    np.save(save_path, result_map)
    print(f"✅ 저장 완료! 파일 경로: {save_path} (크기: {result_map.shape})")

    return result_map


# ==========================================
# 실행을 위한 테스트 코드 추가
# ==========================================
if __name__ == "__main__":
    for map_type in ['SW', 'SE', 'NW', 'NE', 'KWU']:
        generate_enivronment_map(map_type)