import requests
import sqlite3
import time

# 高德 API Key
AMAP_KEY = "6473da94204bc215a7d2f2fb9ce3b2b0"

def geocode(address):
    """调用高德地理编码 API，返回 (lng, lat) 或 (None, None)"""
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {
        "address": address,
        "output": "JSON",
        "key": AMAP_KEY
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        if data["status"] == "1" and data["geocodes"]:
            location = data["geocodes"][0]["location"]
            lng, lat = location.split(",")
            return float(lng), float(lat)
        else:
            print(f"地理编码失败：{address} -> {data.get('info', '未知错误')}")
            return None, None
    except Exception as e:
        print(f"请求异常：{address} -> {e}")
        return None, None

# 连接数据库（根据你的实际路径）
conn = sqlite3.connect("instance/movies.db")
cursor = conn.cursor()

# 获取所有影院（假设表名为 cinema，字段有 id, name, address, lng, lat）
cursor.execute("SELECT id, name, address FROM cinema WHERE lng IS NULL OR lat IS NULL")
cinemas = cursor.fetchall()

print(f"找到 {len(cinemas)} 个待更新影院")
for cid, name, address in cinemas:
    if not address:
        print(f"影院 {name} 无地址，跳过")
        continue
    print(f"正在获取 {name} 坐标...")
    lng, lat = geocode(address)
    if lng is not None:
        cursor.execute("UPDATE cinema SET lng=?, lat=? WHERE id=?", (lng, lat, cid))
        conn.commit()
        print(f"  已更新：{name} -> {lng}, {lat}")
    else:
        print(f"  更新失败：{name}")
    time.sleep(0.2)  # 避免 API 限流

conn.close()
print("所有影院坐标更新完成！")