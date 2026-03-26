import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class HeterogeneousMovieDataset(Dataset):
    def __init__(self, df):
        """
        引入异构特征的数据集
        包含:
        - 用户ID (类别特征)
        - 电影ID (类别特征)
        - 用户年龄 (数值特征)
        - 用户性别 (类别特征 0/1)
        - 电影年份 (数值特征)
        - 电影流派 (类别特征, 假设已转为索引)
        """
        self.user_ids = df['user_idx'].values
        self.movie_ids = df['movie_idx'].values
        self.user_ages = df['user_age'].values.astype(np.float32)
        self.user_genders = df['user_gender'].values
        self.movie_years = df['movie_year'].values.astype(np.float32)
        self.movie_genres = df['movie_genre_idx'].values
        self.ratings = df['rating'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.ratings)
        
    def __getitem__(self, idx):
        return (
            torch.tensor(self.user_ids[idx], dtype=torch.long),
            torch.tensor(self.movie_ids[idx], dtype=torch.long),
            torch.tensor(self.user_ages[idx], dtype=torch.float32),
            torch.tensor(self.user_genders[idx], dtype=torch.long),
            torch.tensor(self.movie_years[idx], dtype=torch.float32),
            torch.tensor(self.movie_genres[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float32)
        )

class HeteroRecommender(nn.Module):
    """
    融合异构特征的深度推荐模型
    """
    def __init__(self, num_users, num_movies, num_genres, embedding_dim=32):
        super(HeteroRecommender, self).__init__()
        
        # ID Embedding
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # 异构分类特征 Embedding
        self.gender_embedding = nn.Embedding(3, 8) # 0:Unknown, 1:Male, 2:Female
        self.genre_embedding = nn.Embedding(num_genres, 16)
        
        # 异构连续特征处理 (Age, Year) - 简单归一化后接入网络，这里在 forward 阶段拼接
        
        # 拼接后的总特征维度: User_Emb(32) + Movie_Emb(32) + Gender_Emb(8) + Genre_Emb(16) + Age(1) + Year(1) = 90
        total_dim = embedding_dim * 2 + 8 + 16 + 2
        
        # MLP 层
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, u_id, m_id, u_age, u_gender, m_year, m_genre):
        u_emb = self.user_embedding(u_id)
        m_emb = self.movie_embedding(m_id)
        gen_emb = self.gender_embedding(u_gender)
        genre_emb = self.genre_embedding(m_genre)
        
        # 对连续特征进行简单的缩放处理以便于训练
        u_age_scaled = (u_age.unsqueeze(1) - 30.0) / 20.0
        m_year_scaled = (m_year.unsqueeze(1) - 2000.0) / 20.0
        
        # 拼接所有特征
        x = torch.cat([u_emb, m_emb, gen_emb, genre_emb, u_age_scaled, m_year_scaled], dim=1)
        
        out = self.mlp(x)
        return out * 10.0 # 输出 0-10 分

def train_hetero_model(df, epochs=10, batch_size=64):
    """
    训练融合了异构特征的模型
    """
    # 构建 ID 映射字典
    user_ids = df['userId'].unique()
    movie_ids = df['movieId'].unique()
    genres = df['primary_genre'].unique()
    
    user2idx = {o: i for i, o in enumerate(user_ids)}
    movie2idx = {o: i for i, o in enumerate(movie_ids)}
    genre2idx = {o: i for i, o in enumerate(genres)}
    
    # 将 DataFrame 转换为索引
    df['user_idx'] = df['userId'].map(user2idx)
    df['movie_idx'] = df['movieId'].map(movie2idx)
    df['movie_genre_idx'] = df['primary_genre'].map(genre2idx)
    
    # 填充缺失值
    df['user_age'] = df['user_age'].fillna(30) # 默认年龄
    # 转换性别 (0:Unknown, 1:Male, 2:Female)
    def map_gender(g):
        if g == 'Male': return 1
        elif g == 'Female': return 2
        return 0
    df['user_gender'] = df['user_gender'].apply(map_gender)
    
    dataset = HeterogeneousMovieDataset(df)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = HeteroRecommender(len(user_ids), len(movie_ids), len(genres))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    print("开始训练异构推荐模型...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u_id, m_id, u_age, u_gen, m_yr, m_gre, ratings in train_loader:
            optimizer.zero_grad()
            preds = model(u_id, m_id, u_age, u_gen, m_yr, m_gre).squeeze()
            
            # 确保维度匹配
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
                
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
    return model, user2idx, movie2idx, genre2idx

# 此处仅为模型结构展示，在实际项目中，可在app.py中导入该模块并在启动或定时任务中训练/更新模型
