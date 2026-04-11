# AI 电影推荐系统

基于 Flask + PyTorch 的电影平台，包含电影浏览、评分收藏、智能推荐、AI 对话、圈子社区和选座购票等功能。

## 主要功能

- 用户注册/登录（支持人脸登录）
- 电影检索、详情页、评论、评分、收藏
- 推荐系统（冷启动 + 内容推荐 + 混合推荐）
- AI 聊天助手（本地规则兜底，可选接入智谱 API，支持站内 RAG 检索）
- 圈子社区（建圈、发帖、评论）
- 影院座位选择与下单（含连坐推荐与优选区价格）
- 数据可视化页面（ECharts）
- TMDB 数据同步（热门、热映、即将上映）

## 技术栈

- 后端：Flask, Flask-SQLAlchemy, Flask-Login, Flask-Migrate
- 算法：PyTorch, sentence-transformers, scikit-learn
- 视觉：OpenCV, Pillow, scikit-image
- 数据：SQLite, pandas, numpy
- 前端：Bootstrap 5, JavaScript, ECharts

## 项目结构

```text
Film/
├── app.py
├── models.py
├── recommender.py
├── content_recommender.py
├── face_auth.py
├── data_loader.py
├── train.py
├── requirements.txt
├── templates/
├── static/
├── migrations/
├── data/
├── models/
└── instance/
```

## 环境要求

- Python 3.10+（建议）
- Windows / Linux / macOS

## 快速开始

```bash
git clone <your-repo-url>
cd Film

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
# source venv/bin/activate

pip install -r requirements.txt
```

启动服务：

```bash
python app.py
```

默认访问地址：`http://127.0.0.1:5000`

## 数据库与命令

首次初始化（若 `instance/movies.db` 不存在）：

```bash
flask init-db
```

常用 CLI：

```bash
flask upgrade-db
flask sync-screenings
flask build-embeddings
flask upgrade-seats
flask init-seats-for-movie <movie_id>
```

## 环境变量（可选）

项目在无额外密钥时也可运行。以下变量用于增强功能：

```env
# 智谱对话模型（可选）
ZHIPU_API_KEY=your_api_key

# Flask 会话密钥（建议在生产环境设置）
SECRET_KEY=your-secret-key
```

## 推荐系统说明

- 冷启动：按电影评分返回热门候选
- 内容推荐：优先使用 `sentence-transformers`，不可用时自动降级到 TF-IDF
- 混合推荐：融合内容相似度与协同过滤预测分
- 支持预计算电影向量并持久化到 `models/movie_embeddings.pkl`

## 依赖说明

完整依赖见 `requirements.txt`，关键依赖包括：

- Flask 及相关扩展（登录、ORM、迁移）
- PyTorch 与推荐算法栈
- OpenCV / Pillow / scikit-image（人脸特征）
- sentence-transformers（语义向量）

## 许可证

MIT
