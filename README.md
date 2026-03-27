# AI 电影推荐系统

基于 Flask 和 PyTorch 构建的智能电影推荐系统，集成了用户认证、电影展示、数据可视化、个性化推荐等功能。

## 功能特性

- **用户系统**：注册、登录、个人中心管理，支持修改邮箱、年龄、性别等信息
- **电影浏览**：关键词搜索、分页浏览，展示电影海报、类型、评分等详细信息
- **电影详情**：查看电影详情、发表评论、添加收藏、评分功能
- **收藏管理**：个人收藏夹，支持添加和移除电影
- **数据可视化**：基于 ECharts 的数据大屏，展示年度发布趋势、流派占比、评分分布
- **智能推荐**：
  - 冷启动推荐：高分热门电影
  - 内容推荐：基于 BERT/TF-IDF 语义相似度
  - 混合推荐：融合内容相似度与深度学习预测

## 技术栈

| 类别 | 技术 |
|------|------|
| 后端 | Python 3.8+, Flask, Flask-SQLAlchemy, Flask-Login |
| 前端 | HTML5, CSS3, JavaScript, Bootstrap 5, ECharts |
| 深度学习 | PyTorch, Sentence-Transformers, Scikit-learn |
| 数据库 | SQLite (开发环境) |
| 数据源 | TMDB API |

## 项目结构

```
Film/
├── app.py                 # Flask 主程序及路由
├── models.py              # 数据库模型定义
├── recommender.py         # 深度学习推荐模型
├── content_recommender.py # 内容推荐引擎
├── data_loader.py         # TMDB 数据加载器
├── train.py               # 模型训练脚本
├── requirements.txt       # 依赖包列表
├── templates/             # HTML 模板
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── movies.html
│   ├── movie_detail.html
│   ├── favorites.html
│   ├── recommendations.html
│   ├── profile.html
│   └── visualize.html
├── models/                # 训练好的模型文件
│   ├── recommender.pth
│   ├── movie_embeddings.pkl
│   └── mappings.pkl
├── data/                  # 数据文件
└── instance/              # SQLite 数据库
```

## 快速开始

### 1. 环境准备

确保已安装 Python 3.8 或更高版本。

```bash
# 克隆项目
git clone https://github.com/your-username/Film.git
cd Film

# 创建虚拟环境 (推荐)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

> 注意：`torch`（PyTorch）体积约 2GB，安装耗时较长，请耐心等待。

### 2. 启动服务

项目已内置 `instance/movies.db` 数据库，**无需初始化**，直接运行：

```bash
python app.py
```

访问 [http://127.0.0.1:5000](http://127.0.0.1:5000)，注册账号即可开始使用。

### 3. 构建内容向量（可选，用于个性化推荐）

```bash
flask build-embeddings
```

此命令会使用 BERT 为所有电影计算语义向量，首次运行需下载语言模型（需联网）。  
跳过此步骤，推荐功能将退化为热门推荐，其他功能完全不受影响。

### 4. 训练推荐模型（可选）

```bash
python train.py
```

## 推荐算法说明

### 深度学习模型 (HeteroRecommender)

融合异构特征的神经网络推荐模型，包含：

- **用户特征**：ID Embedding、年龄、性别
- **电影特征**：ID Embedding、年份、流派
- **网络结构**：Embedding 层 → 特征拼接 → MLP → 评分预测

### 内容推荐

- 使用 Sentence-Transformers (BERT) 提取电影语义向量
- 基于用户高分电影构建偏好向量
- 计算余弦相似度推荐相似电影

## 数据模型

| 模型 | 说明 |
|------|------|
| User | 用户信息 (用户名、密码哈希、邮箱、年龄、性别) |
| Movie | 电影信息 (标题、简介、年份、流派、评分、海报) |
| Rating | 用户评分记录 |
| Favorite | 用户收藏记录 |
| Comment | 用户评论 |

## 配置说明

在 `app.py` 中可修改以下配置：

```python
app.config['SECRET_KEY'] = 'your-secret-key'  # 会话密钥
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movies.db'  # 数据库路径
```

TMDB API 密钥配置在项目根目录的 `.env` 文件中（参考 `.env.example`）：

```
TMDB_API_KEY=your-tmdb-api-key
SECRET_KEY=your-secret-key
```

> 项目已内置数据库，正常使用无需配置 TMDB API 密钥。

## 依赖说明

主要依赖包：

- Flask 3.0.0 - Web 框架
- Flask-SQLAlchemy 3.1.1 - ORM
- Flask-Login 0.6.3 - 用户认证
- PyTorch 2.1.2 - 深度学习框架
- sentence-transformers - BERT 语义向量
- scikit-learn 1.3.2 - TF-IDF 与相似度计算
- pandas 2.1.4 - 数据处理

## 许可证

MIT License
