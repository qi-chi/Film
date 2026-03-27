import os
import random
import urllib.parse
import pickle
import torch
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Movie, Favorite, Comment, Rating
from recommender import HeteroRecommender
import uuid
from werkzeug.utils import secure_filename
from content_recommender import (
    compute_and_save_embeddings, get_hybrid_recommendations, embeddings_ready
)

app = Flask(__name__)

# 头像上传配置
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads', 'avatars')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

app.config['SECRET_KEY'] = 'your-secret-key-for-development'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movies.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '请先登录以访问此页面。'
from flask_migrate import Migrate
migrate = Migrate(app, db)
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- CLI Commands ---

@app.cli.command("build-embeddings")
def build_embeddings_cmd():
    """预计算所有电影的内容向量（BERT 或 TF-IDF），用于内容推荐。"""
    with app.app_context():
        movies = Movie.query.all()
        if not movies:
            print("数据库中没有电影，请先运行 flask init-db")
            return
        print(f"正在为 {len(movies)} 部电影计算内容向量...")
        compute_and_save_embeddings(movies)

@app.cli.command("init-db")
def init_db():
    db.create_all()
    # 导入 TMDB 数据
    if Movie.query.count() == 0:
        print("正在从 TMDB 获取最新热门电影数据，这可能需要几十秒钟...")
        from data_loader import fetch_tmdb_popular_movies
        
        # 获取 xx 部热门电影
        movies_df = fetch_tmdb_popular_movies(1000)#获取xx部电影
        
        count = 0
        for _, row in movies_df.iterrows():
            m = Movie(
                title=row['title'],
                description=row['overview'],
                release_year=row['year'],
                genre=row['primary_genre'],
                rating=row['rating'],
                poster_url=row['poster_url'] or f"https://via.placeholder.com/300x450?text={urllib.parse.quote(row['title'][:10])}"
            )
            db.session.add(m)
            count += 1
        
        db.session.commit()
        print(f"数据库初始化完成，成功导入 {count} 部 TMDB 中文电影数据。")

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

# 1. 注册与登录
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('用户名已存在，请选择其他用户名。', 'danger')
            return redirect(url_for('register'))
            
        user = User(
            username=username,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        flash('注册成功，请登录。', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('登录成功！', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('movies'))
        else:
            flash('用户名或密码错误。', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('您已退出登录。', 'info')
    return redirect(url_for('index'))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload_avatar', methods=['POST'])
@login_required
def upload_avatar():
    # 检查是否有文件上传
    if 'avatar' not in request.files:
        return jsonify({'error': '未选择文件'}), 400
    file = request.files['avatar']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件类型，请上传 jpg/png/gif 图片'}), 400

    # 生成安全的文件名（避免重名）
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{current_user.id}_{uuid.uuid4().hex}.{ext}"
    # 保存文件
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # 保存相对路径到数据库（用于前端显示）
    avatar_url = url_for('static', filename=f'uploads/avatars/{filename}')
    current_user.avatar = avatar_url
    db.session.commit()

    return jsonify({'success': True, 'avatar_url': avatar_url})

# 7. 个人中心
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        age = request.form.get('age')
        gender = request.form.get('gender')
        
        # 处理用户名修改（唯一性校验）
        if username and username != current_user.username:
            # 检查用户名是否已被其他用户占用
            existing_user = User.query.filter(User.username == username, User.id != current_user.id).first()
            if existing_user:
                flash('用户名已被占用，请选择其他用户名。', 'danger')
                return redirect(url_for('profile'))
            current_user.username = username
        
        # 更新其他信息
        if email:
    # 格式校验（简单正则）
            import re
            email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_regex, email):
                flash('邮箱格式无效，请重新输入。', 'danger')
                return redirect(url_for('profile'))
            current_user.email = email
        else:
    # 用户清空邮箱，设为 None
            current_user.email = None
        if age:
            try:
                current_user.age = int(age)
            except ValueError:
                flash('年龄必须是数字。', 'danger')
                return redirect(url_for('profile'))
        else:
            current_user.age = None
        current_user.gender = gender
        
        db.session.commit()
        flash('个人信息更新成功。', 'success')
        return redirect(url_for('profile'))
        
    return render_template('profile.html')
# 2. 电影展示与搜索
@app.route('/movies')
@login_required
def movies():
    page = request.args.get('page', 1, type=int)
    search_query = request.args.get('q', '')
    genre_filter = request.args.get('genre', '')
    
    query = Movie.query
    if search_query:
        query = query.filter(Movie.title.ilike(f'%{search_query}%'))
    if genre_filter:
        query = query.filter(Movie.genre == genre_filter)
        
    pagination = query.paginate(page=page, per_page=12, error_out=False)
    movies = pagination.items
    
    # 获取所有可用的电影分类供前端展示
    all_genres = [g[0] for g in db.session.query(Movie.genre).distinct().all() if g[0]]
    
    return render_template('movies.html', movies=movies, pagination=pagination, search_query=search_query, current_genre=genre_filter, all_genres=all_genres)

# 3. 电影详情页
@app.route('/movie/<int:movie_id>', methods=['GET', 'POST'])
@login_required
def movie_detail(movie_id):
    movie = Movie.query.get_or_404(movie_id)
    is_favorite = Favorite.query.filter_by(user_id=current_user.id, movie_id=movie_id).first() is not None
    
    if request.method == 'POST':
        content = request.form.get('content')
        if content:
            comment = Comment(user_id=current_user.id, movie_id=movie_id, content=content)
            db.session.add(comment)
            db.session.commit()
            flash('评论发表成功！', 'success')
            return redirect(url_for('movie_detail', movie_id=movie_id))
            
    comments = Comment.query.filter_by(movie_id=movie_id).order_by(Comment.created_at.desc()).all()
    user_rating = Rating.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
    return render_template('movie_detail.html', movie=movie, is_favorite=is_favorite,
                           comments=comments, user_rating=user_rating)

# 4. 收藏列表
@app.route('/toggle_favorite/<int:movie_id>', methods=['POST'])
@login_required
def toggle_favorite(movie_id):
    movie = Movie.query.get_or_404(movie_id)
    favorite = Favorite.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
    
    if favorite:
        db.session.delete(favorite)
        db.session.commit()
        return jsonify({'status': 'removed'})
    else:
        new_fav = Favorite(user_id=current_user.id, movie_id=movie_id)
        db.session.add(new_fav)
        db.session.commit()
        return jsonify({'status': 'added'})

@app.route('/favorites')
@login_required
def favorites():
    favs = Favorite.query.filter_by(user_id=current_user.id).all()
    movies = [f.movie for f in favs]
    return render_template('favorites.html', movies=movies)

# 5. 数据可视化
@app.route('/visualize')
@login_required
def visualize():
    return render_template('visualize.html')

# API for Data Visualization
@app.route('/api/chart-data')
@login_required
def chart_data():
    # 模拟从数据库获取各种维度的统计数据
    # 比如按年份统计电影数量
    year_stats = db.session.query(Movie.release_year, db.func.count(Movie.id)).group_by(Movie.release_year).all()
    years = [str(y[0]) for y in year_stats if y[0]]
    year_counts = [y[1] for y in year_stats if y[0]]
    
    # 比如按流派统计
    genre_stats = db.session.query(Movie.genre, db.func.count(Movie.id)).group_by(Movie.genre).all()
    genres = [{'value': g[1], 'name': g[0]} for g in genre_stats if g[0]]
    
    return jsonify({
        'years': years,
        'year_counts': year_counts,
        'genres': genres
    })

# 6. 电影推荐（混合引擎：冷启动 / 内容推荐 / NCF 融合）
@app.route('/recommendations')
@login_required
def recommendations():
    all_movies = Movie.query.all()
    user_ratings = Rating.query.filter_by(user_id=current_user.id).all()

    # ── 构建 NCF 评分器（若模型文件存在）──
    ncf_scorer = None
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'recommender.pth')
    dict_path  = os.path.join(os.path.dirname(__file__), 'models', 'mappings.pkl')

    if os.path.exists(model_path) and os.path.exists(dict_path):
        try:
            with open(dict_path, 'rb') as f:
                mappings = pickle.load(f)

            user2idx  = mappings['user2idx']
            movie2idx = mappings['movie2idx']
            genre2idx = mappings['genre2idx']

            ncf_model = HeteroRecommender(len(user2idx), len(movie2idx), len(genre2idx))
            ncf_model.load_state_dict(
                torch.load(model_path, map_location='cpu', weights_only=True)
            )
            ncf_model.eval()

            u_id     = current_user.id % len(user2idx) if user2idx else 0
            u_age    = float(current_user.age or 30)
            u_gender = 1 if current_user.gender == 'Male' else (2 if current_user.gender == 'Female' else 0)

            def ncf_scorer(movies):
                if not movies:
                    return []
                m_ids   = torch.tensor([movie2idx.get(m.id, 0) for m in movies], dtype=torch.long)
                m_years = torch.tensor([float(m.release_year or 2000) for m in movies], dtype=torch.float32)
                m_gens  = torch.tensor([genre2idx.get(m.genre, 0) for m in movies], dtype=torch.long)
                bs      = len(movies)
                u_ids   = torch.tensor([u_id]     * bs, dtype=torch.long)
                u_ages  = torch.tensor([u_age]    * bs, dtype=torch.float32)
                u_gens  = torch.tensor([u_gender] * bs, dtype=torch.long)
                with torch.no_grad():
                    preds = ncf_model(u_ids, m_ids, u_ages, u_gens, m_years, m_gens).squeeze()
                if preds.dim() == 0:
                    preds = preds.unsqueeze(0)
                return preds.tolist()
        except Exception:
            ncf_scorer = None

    # ── 调用混合推荐引擎，至少返回 10 部 ──
    rec_pairs, rec_type = get_hybrid_recommendations(
        user_ratings, all_movies, ncf_scorer=ncf_scorer, n=10
    )

    # 检查内容向量是否就绪，供前端提示
    emb_ready = embeddings_ready()

    return render_template(
        'recommendations.html',
        rec_pairs=rec_pairs,
        rec_type=rec_type,
        emb_ready=emb_ready,
        rating_count=len(user_ratings)
    )


# API：返回推荐 JSON（供前端 AJAX 刷新）
@app.route('/api/recommendations')
@login_required
def api_recommendations():
    return recommendations()  # 复用同一逻辑（实际项目可拆分为独立函数）

# 提交/修改评分（范围 1~10，支持半星 step=0.5）
@app.route('/rate_movie/<int:movie_id>', methods=['POST'])
@login_required
def rate_movie(movie_id):
    score = request.form.get('score', type=float)
    if score is not None and 1.0 <= score <= 10.0:
        existing = Rating.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
        if existing:
            existing.score = score
            flash(f'评分已更新为 {score} 分！', 'success')
        else:
            db.session.add(Rating(user_id=current_user.id, movie_id=movie_id, score=score))
            flash(f'评分 {score} 分提交成功！每一次打分都让推荐更懂你。', 'success')
        db.session.commit()
    return redirect(url_for('movie_detail', movie_id=movie_id))


# 删除评分
@app.route('/delete_rating/<int:movie_id>', methods=['POST'])
@login_required
def delete_rating(movie_id):
    rating = Rating.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
    if rating:
        db.session.delete(rating)
        db.session.commit()
        return jsonify({'status': 'deleted'})
    return jsonify({'status': 'not_found'}), 404

# 8. AI 小助手 API（智谱 GLM 接入 · 纯文本无任何格式）
@app.route('/api/chat', methods=['POST'])
@login_required
def ai_chat():
    try:
        from zhipuai import ZhipuAI
    except:
        return jsonify({'reply': 'AI 服务未就绪，请检查依赖包'})
    
    data = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'reply': '请输入你想咨询的问题~'})

    try:
        # ========== 智谱 AI 环境变量版本（可直接提交Git）==========
        from zhipuai import ZhipuAI

# 从系统环境变量读取 API Key，没有则不启用AI（不报错）
        API_KEY = os.getenv("ZHIPU_API_KEY", "")

        client = None
        if API_KEY:
            try:
                client = ZhipuAI(api_key=API_KEY)
            except:
                client = None
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {
                    "role": "system",
                    "content": "你是电影推荐助手，只能用纯自然口语中文回答，绝对禁止使用任何 Markdown 格式，禁止使用 ## ** - 等任何符号，不要标题，不要列表，不要加粗，只用正常段落文字回答。"
                },
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
        )
        
        reply = response.choices[0].message.content.strip()
        
        # 强制清理所有格式符号，确保 100% 干净文本
        reply = reply.replace("**", "")
        reply = reply.replace("## ", "")
        reply = reply.replace("- ", "")
        reply = reply.replace("### ", "")
        reply = reply.replace("* ", "")
        reply = reply.replace("> ", "")

    except Exception as e:
        reply = "AI 小助手暂时无法服务，请稍后再试"

    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
