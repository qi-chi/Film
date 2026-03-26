import os
import torch
import pickle
import pandas as pd
import numpy as np
from recommender import train_hetero_model
from app import app, db
from models import User, Movie, Rating

def get_real_ratings():
    """
    从 SQLite 数据库中获取真实用户的评分记录和异构特征
    """
    with app.app_context():
        ratings = Rating.query.all()
        if len(ratings) < 10:  # 设定一个阈值，如果真实数据太少，可能无法有效训练
            return None
            
        print(f"检测到 {len(ratings)} 条真实的数据库评分记录。")
        ratings_data = []
        for r in ratings:
            user = r.user
            movie = r.movie
            
            ratings_data.append({
                'userId': user.id,
                'user_age': user.age if user.age else 30,
                'user_gender': user.gender if user.gender else 'Unknown',
                'movieId': movie.id,
                'movie_year': movie.release_year if movie.release_year else 2000,
                'primary_genre': movie.genre if movie.genre else 'Unknown',
                'rating': r.score
            })
            
        return pd.DataFrame(ratings_data)

def generate_mock_ratings(movies_df, num_users=500, min_ratings_per_user=10, max_ratings_per_user=30):
    """
    生成模拟的异构用户评分数据 (用于冷启动)
    """
    print(f"由于真实数据不足，正在生成 {num_users} 个模拟用户的评分数据和异构特征以供冷启动...")
    
    # 这里的 movies_df 必须包含 id 而不仅仅是 tmdb_id，因为我们需要与数据库对齐
    # 但传进来的如果是刚爬取的数据，用 tmdb_id 也可以，但最好从数据库直接拿
    with app.app_context():
        db_movies = Movie.query.all()
        if not db_movies:
            print("错误：数据库中没有电影数据，请先运行 flask init-db")
            return pd.DataFrame()
            
        movie_features = {
            m.id: {
                'year': m.release_year or 2000,
                'primary_genre': m.genre or 'Unknown'
            } for m in db_movies
        }
        movie_ids = list(movie_features.keys())
    
    ratings_data = []
    
    user_profiles = {}
    for uid in range(1, num_users + 1):
        user_profiles[uid] = {
            'age': np.random.randint(18, 60),
            'gender': np.random.choice(['Male', 'Female', 'Unknown'])
        }
    
    for user_id in range(1, num_users + 1):
        num_ratings = np.random.randint(min_ratings_per_user, max_ratings_per_user + 1)
        # 确保不会抽取超过电影总数
        actual_num = min(num_ratings, len(movie_ids))
        if actual_num == 0: continue
            
        rated_movies = np.random.choice(movie_ids, size=actual_num, replace=False)
        
        for mid in rated_movies:
            rating = np.random.randint(1, 11) / 2.0
            m_feat = movie_features[mid]
            
            ratings_data.append({
                'userId': user_id,
                'user_age': user_profiles[user_id]['age'],
                'user_gender': user_profiles[user_id]['gender'],
                'movieId': mid,
                'movie_year': m_feat['year'],
                'primary_genre': m_feat['primary_genre'],
                'rating': rating
            })
            
    return pd.DataFrame(ratings_data)

def main():
    print("正在准备数据集...")
    
    # 尝试获取真实数据
    ratings_df = get_real_ratings()
    
    # 如果真实数据不够，使用模拟数据冷启动
    if ratings_df is None or len(ratings_df) < 50:
        print("提示：真实评分数据不足，将使用模拟数据进行冷启动训练。当用户产生更多评分后，可再次运行此脚本以使用真实数据训练。")
        ratings_df = generate_mock_ratings(None) # 已经改为了从数据库取电影特征
        
    if ratings_df.empty:
        print("没有可用的数据进行训练。")
        return
        
    print(f"开始使用 {len(ratings_df)} 条评分数据训练异构模型。")
    
    epochs = 10
    batch_size = 64
    model, user2idx, movie2idx, genre2idx = train_hetero_model(ratings_df, epochs=epochs, batch_size=batch_size)
    
    print("正在保存模型和字典...")
    os.makedirs("models", exist_ok=True)
    
    torch.save(model.state_dict(), "models/recommender.pth")
    
    mappings = {
        'user2idx': user2idx,
        'movie2idx': movie2idx,
        'genre2idx': genre2idx,
        'idx2movie': {v: k for k, v in movie2idx.items()}
    }
    with open("models/mappings.pkl", "wb") as f:
        pickle.dump(mappings, f)
        
    print("训练完成！异构特征推荐模型已保存在 models/ 目录下。")

if __name__ == "__main__":
    main()
