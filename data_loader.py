import os
import pandas as pd
import requests
import time

# 自己申请的 API 密钥
API_KEY = "a57e41d26308cc64a5fc07fe9c43944e"

def get_tmdb_genres():
    """
    获取 TMDB 中文电影流派映射字典
    """
    url = f"https://api.themoviedb.org/3/genre/movie/list"
    params = {'api_key': API_KEY, 'language': 'zh-CN'}
    
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            genres = response.json().get('genres', [])
            return {g['id']: g['name'] for g in genres}
    except Exception as e:
        print(f"获取流派列表失败: {e}")
    return {}

def fetch_tmdb_popular_movies(total_movies=100):
    """
    直接从 TMDB 获取热门电影列表，不再使用 MovieLens
    返回包含电影信息的 DataFrame
    """
    genre_map = get_tmdb_genres()
    movies = []
    
    # TMDB 每页 20 条数据，计算需要请求多少页
    pages_needed = (total_movies // 20) + (1 if total_movies % 20 != 0 else 0)
    
    url = "https://api.themoviedb.org/3/movie/popular"
    
    for page in range(1, pages_needed + 1):
        params = {
            'api_key': API_KEY,
            'language': 'zh-CN',
            'page': page
        }
        
        try:
            print(f"正在从 TMDB 拉取第 {page} 页电影数据...")
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                results = response.json().get('results', [])
                for item in results:
                    if len(movies) >= total_movies:
                        break
                        
                    # 获取主类型中文名
                    genre_ids = item.get('genre_ids', [])
                    primary_genre = genre_map.get(genre_ids[0], "未知") if genre_ids else "未知"
                    
                    # 提取年份
                    release_date = item.get('release_date', '')
                    year = int(release_date.split('-')[0]) if release_date else 2000
                    
                    poster_path = item.get('poster_path')
                    
                    movie_data = {
                        'tmdb_id': item['id'],
                        'title': item.get('title', '未知标题'),
                        'overview': item.get('overview', '暂无中文简介。'),
                        'year': year,
                        'primary_genre': primary_genre,
                        'rating': item.get('vote_average', 0.0), # TMDB 是 10分制
                        'poster_url': f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
                    }
                    movies.append(movie_data)
            time.sleep(0.1) # 防封
        except Exception as e:
            print(f"请求 TMDB 失败: {e}")
            break
            
    return pd.DataFrame(movies)

def _fetch_tmdb_movies(endpoint, total_movies=100, type_label=''):
    """
    通用 TMDB 电影列表拉取函数。
    endpoint: TMDB 路径，如 /movie/now_playing
    """
    genre_map = get_tmdb_genres()
    movies = []
    pages_needed = (total_movies // 20) + (1 if total_movies % 20 != 0 else 0)
    url = f"https://api.themoviedb.org/3{endpoint}"

    for page in range(1, pages_needed + 1):
        params = {'api_key': API_KEY, 'language': 'zh-CN', 'region': 'CN', 'page': page}
        try:
            print(f"正在拉取 [{type_label}] 第 {page} 页（地区：CN）...")
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                results = response.json().get('results', [])
                for item in results:
                    if len(movies) >= total_movies:
                        break
                    genre_ids = item.get('genre_ids', [])
                    primary_genre = genre_map.get(genre_ids[0], '未知') if genre_ids else '未知'
                    release_date = item.get('release_date', '')
                    year = int(release_date.split('-')[0]) if release_date else 2000
                    poster_path = item.get('poster_path')
                    movies.append({
                        'tmdb_id': item['id'],
                        'title': item.get('title', '未知标题'),
                        'overview': item.get('overview', '暂无中文简介。'),
                        'year': year,
                        'release_date': release_date,
                        'primary_genre': primary_genre,
                        'rating': item.get('vote_average', 0.0),
                        'poster_url': f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None,
                    })
            time.sleep(0.1)
        except Exception as e:
            print(f"请求 TMDB 失败: {e}")
            break

    return pd.DataFrame(movies)


def fetch_tmdb_now_playing(total_movies=100):
    """从 TMDB 获取正在热映电影列表（/movie/now_playing）"""
    return _fetch_tmdb_movies('/movie/now_playing', total_movies, type_label='正在热映')


def fetch_tmdb_upcoming(total_movies=100):
    """从 TMDB 获取即将上映电影列表（/movie/upcoming）"""
    return _fetch_tmdb_movies('/movie/upcoming', total_movies, type_label='即将上映')


if __name__ == "__main__":
    df = fetch_tmdb_popular_movies(10)
    print(f"成功获取了 {len(df)} 部 TMDB 电影。")
    print(df[['title', 'primary_genre', 'year']])
