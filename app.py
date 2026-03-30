import os
import random
import urllib.parse
import pickle
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Movie, Favorite, Comment, Rating
import uuid
from werkzeug.utils import secure_filename
from face_auth import extract_face_encoding, compare_faces, find_best_match, encoding_to_json

# 尝试导入 PyTorch 相关模块
try:
    import torch
    from recommender import HeteroRecommender
    from content_recommender import (
        compute_and_save_embeddings, get_hybrid_recommendations, embeddings_ready
    )
    pytorch_available = True
except ImportError:
    print("⚠️  PyTorch 导入失败，推荐系统将使用基础模式")
    pytorch_available = False

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
from flask_migrate import Migrate  # pyright: ignore[reportMissingModuleSource]
migrate = Migrate(app, db)
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- CLI Commands ---

@app.cli.command("upgrade-db")
def upgrade_db():
    """自动执行数据库迁移，确保所有模型字段都同步到数据库。"""
    from flask_migrate import upgrade
    with app.app_context():
        upgrade()
    print("数据库迁移执行完成！")

@app.cli.command("build-embeddings")
def build_embeddings_cmd():
    """预计算所有电影的内容向量（BERT 或 TF-IDF），用于内容推荐。"""
    if not pytorch_available:
        print("⚠️  PyTorch 不可用，无法构建内容向量")
        return
    
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
    print("数据库初始化完成！")

# --- 路由定义 ---

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/home')
@login_required
def home():
    # 首页展示逻辑
    return render_template('index.html')

# 1. 认证相关

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('登录成功！', 'success')
            return redirect(url_for('home'))
        else:
            flash('用户名或密码错误', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        face_encoding = request.form.get('face_encoding')
        
        if not username or not password:
            flash('请填写所有必填字段', 'error')
            return redirect(url_for('register'))
        
        # 检查用户名是否已存在
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('用户名已被使用', 'error')
            return redirect(url_for('register'))
        
        # 创建新用户
        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            face_encoding=face_encoding
        )
        db.session.add(user)
        db.session.commit()
        
        flash('注册成功！请登录', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# 人脸登录相关路由

@app.route('/face_login')
def face_login_page():
    """人脸识别登录页面"""
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return render_template('face_login.html')

@app.route('/api/face/login', methods=['POST'])
def face_login():
    """
    人脸识别登录 API
    接收前端传来的 base64 图片，进行人脸比对
    """
    if current_user.is_authenticated:
        return jsonify({'success': False, 'message': '您已登录'})
    
    data = request.get_json()
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False, 'message': '未接收到图片数据'})
    
    # 提取人脸特征
    success, result = extract_face_encoding(image_data)
    
    if not success:
        return jsonify({'success': False, 'message': result})
    
    unknown_encoding = result
    
    # 获取所有已注册人脸的用户
    users_with_face = User.query.filter(User.face_encoding.isnot(None)).all()
    
    if not users_with_face:
        return jsonify({'success': False, 'message': '系统中暂无注册用户人脸数据'})
    
    # 构建用户特征列表
    user_encodings = []
    for user in users_with_face:
        user_encodings.append({
            'id': user.id,
            'username': user.username,
            'face_encoding': user.face_encoding
        })
    
    # 查找最佳匹配
    best_match, distance, debug_info = find_best_match(unknown_encoding, user_encodings)
    
    print(f"人脸登录调试信息: {debug_info}")
    
    if best_match:
        # 登录成功
        user = User.query.get(best_match['id'])
        login_user(user)
        return jsonify({
            'success': True, 
            'message': f'欢迎回来，{user.username}！',
            'username': user.username,
            'distance': round(distance, 4),
            'debug': debug_info,
            'redirect_url': url_for('home')
        })
    else:
        return jsonify({
            'success': False, 
            'message': '人脸识别失败，未找到匹配用户，请尝试密码登录',
            'debug': debug_info
        })

@app.route('/api/face/register', methods=['POST'])
def face_register():
    """
    注册/更新人脸数据 API
    如果用户已登录，更新人脸特征；如果未登录，只返回人脸特征
    """
    data = request.get_json()
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False, 'message': '未接收到图片数据'})
    
    # 提取人脸特征
    success, result = extract_face_encoding(image_data)
    
    if not success:
        return jsonify({'success': False, 'message': result})
    
    # 如果用户已登录，保存到当前用户
    if current_user.is_authenticated:
        encoding_json = encoding_to_json(result)
        current_user.face_encoding = encoding_json
        db.session.commit()
        return jsonify({
            'success': True, 
            'message': '人脸数据注册成功，下次可以使用人脸识别登录'
        })
    else:
        # 用户未登录，返回人脸特征供注册表单使用
        encoding_json = encoding_to_json(result)
        return jsonify({
            'success': True, 
            'encoding': encoding_json,
            'message': '人脸特征提取成功'
        })

@app.route('/api/face/check', methods=['GET'])
@login_required
def check_face_registered():
    """检查当前用户是否已注册人脸"""
    has_face = current_user.face_encoding is not None
    return jsonify({
        'has_face': has_face,
        'username': current_user.username
    })

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload_avatar', methods=['POST'])
@login_required
def upload_avatar():
    """上传头像 API"""
    if 'avatar' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    
    file = request.files['avatar']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # 生成唯一文件名
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 保存到用户信息
        current_user.avatar = filename
        db.session.commit()
        
        return jsonify({
            'success': True, 
            'message': '头像上传成功',
            'avatar_url': url_for('static', filename='uploads/avatars/' + filename)
        })
    else:
        return jsonify({'success': False, 'message': '不支持的文件类型'})

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """用户个人资料页面"""
    if request.method == 'POST':
        # 处理个人信息更新
        email = request.form.get('email')
        age = request.form.get('age')
        gender = request.form.get('gender')
        bio = request.form.get('bio')
        
        current_user.email = email if email else None
        current_user.age = age if age else None
        current_user.gender = gender if gender else None
        current_user.bio = bio if bio else None
        
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
    
    # 协同过滤推荐：喜欢这个电影的人同样也喜欢的电影
    similar_movies = []
    
    # 1. 找到喜欢这个电影的所有用户（收藏或评分 >= 7 的用户）
    users_who_liked = set()
    
    # 从收藏表获取用户
    favorites = Favorite.query.filter_by(movie_id=movie_id).all()
    users_who_liked.update([f.user_id for f in favorites])
    
    # 从评分表获取高评分用户（评分 >= 7）
    high_ratings = Rating.query.filter_by(movie_id=movie_id).filter(Rating.score >= 7).all()
    users_who_liked.update([r.user_id for r in high_ratings])
    
    # 2. 找到这些用户还喜欢的其他电影
    if users_who_liked:
        # 统计每个电影被这些用户喜欢的次数
        movie_like_count = {}
        
        # 从收藏表统计
        other_favorites = Favorite.query.filter(Favorite.user_id.in_(users_who_liked)).all()
        for fav in other_favorites:
            if fav.movie_id != movie_id:  # 排除当前电影
                movie_like_count[fav.movie_id] = movie_like_count.get(fav.movie_id, 0) + 1
        
        # 从评分表统计（评分 >= 7）
        other_ratings = Rating.query.filter(Rating.user_id.in_(users_who_liked)).filter(Rating.score >= 7).all()
        for rating in other_ratings:
            if rating.movie_id != movie_id:  # 排除当前电影
                movie_like_count[rating.movie_id] = movie_like_count.get(rating.movie_id, 0) + 1
        
        # 3. 按被喜欢次数排序，取前 6 个
        sorted_movies = sorted(movie_like_count.items(), key=lambda x: x[1], reverse=True)[:6]
        
        # 4. 获取电影详情
        for movie_id_like, count in sorted_movies:
            m = Movie.query.get(movie_id_like)
            if m:
                similar_movies.append({
                    'movie': m,
                    'like_count': count
                })
    
    return render_template('movie_detail.html', movie=movie, is_favorite=is_favorite,
                           comments=comments, user_rating=user_rating, similar_movies=similar_movies)

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

@app.route('/my_comments')
@login_required
def my_comments():
    comments = Comment.query.filter_by(user_id=current_user.id).order_by(Comment.created_at.desc()).all()
    return render_template('my_comments.html', comments=comments)

@app.route('/delete_comment/<int:comment_id>', methods=['POST'])
@login_required
def delete_comment(comment_id):
    comment = Comment.query.get_or_404(comment_id)
    if comment.user_id != current_user.id:
        return jsonify({'success': False, 'error': '无权删除此评论'})
    
    db.session.delete(comment)
    db.session.commit()
    return jsonify({'success': True})

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

    if not pytorch_available:
        # PyTorch 不可用时，返回随机推荐
        random.shuffle(all_movies)
        rec_pairs = [(movie, 0.0) for movie in all_movies[:10]]
        rec_type = "随机推荐"
        emb_ready = False
    else:
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
    return jsonify({'status': 'error'})

if __name__ == '__main__':
    app.run(debug=True)