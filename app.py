import os
import random
import urllib.parse
import pickle
import torch
import re
import uuid
import click  # 添加这行导入
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Movie, Favorite, Comment, Rating
from recommender import HeteroRecommender

from werkzeug.utils import secure_filename
from content_recommender import (
    compute_and_save_embeddings, get_hybrid_recommendations, embeddings_ready
)
from datetime import datetime

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
    """为现有数据库添加 tmdb_id / movie_type / release_date 字段（幂等，可重复执行）。"""
    import sqlite3
    db_path = os.path.join(os.path.dirname(__file__), 'instance', 'movies.db')
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    new_cols = [
        ("tmdb_id",      "INTEGER"),
        ("movie_type",   "VARCHAR(20) DEFAULT 'popular'"),
        ("release_date", "VARCHAR(20)"),
    ]
    for col_name, col_def in new_cols:
        try:
            cur.execute(f"ALTER TABLE movie ADD COLUMN {col_name} {col_def}")
            print(f"  已添加字段: {col_name}")
        except Exception:
            print(f"  字段已存在（跳过）: {col_name}")
    conn.commit()
    conn.close()
    print("数据库升级完成。")


@app.cli.command("sync-screenings")
def sync_screenings():
    """从 TMDB 同步正在热映和即将上映电影到数据库（按 tmdb_id 去重）。"""
    from data_loader import fetch_tmdb_now_playing, fetch_tmdb_upcoming

    with app.app_context():
        existing = {
            m.tmdb_id: m
            for m in Movie.query.filter(Movie.tmdb_id.isnot(None)).all()
        }

        def upsert_batch(df, movie_type):
            from datetime import date
            today = date.today().isoformat()
            added = updated = skipped = 0
            for _, row in df.iterrows():
                # upcoming 模式下，跳过上映日期已过期的条目
                rd = row.get('release_date', '') or ''
                if movie_type == 'upcoming' and rd and rd < today:
                    skipped += 1
                    continue
                tid = int(row['tmdb_id'])
                poster = row['poster_url'] or \
                    f"https://via.placeholder.com/300x450?text={urllib.parse.quote(row['title'][:10])}"
                if tid in existing:
                    m = existing[tid]
                    m.movie_type   = movie_type
                    m.release_date = rd
                    updated += 1
                else:
                    m = Movie(
                        tmdb_id      = tid,
                        title        = row['title'],
                        description  = row['overview'],
                        release_year = row['year'],
                        release_date = rd,
                        genre        = row['primary_genre'],
                        rating       = row['rating'],
                        poster_url   = poster,
                        movie_type   = movie_type,
                    )
                    db.session.add(m)
                    existing[tid] = m
                    added += 1
            db.session.commit()
            print(f"  [{movie_type}] 新增 {added} 部，更新 {updated} 部，跳过过期 {skipped} 部。")

        print("正在拉取正在热映电影...")
        now_df = fetch_tmdb_now_playing(100)
        if not now_df.empty:
            upsert_batch(now_df, 'now_playing')

        print("正在拉取即将上映电影...")
        up_df = fetch_tmdb_upcoming(100)
        if not up_df.empty:
            upsert_batch(up_df, 'upcoming')

        print("同步完成！")


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
    # ✅ 添加：判断电影是热映还是待映
    from datetime import date
    today = date.today().isoformat()
    if movie.movie_type == 'now_playing' or (movie.release_date and movie.release_date <= today):
        movie.tag = "now"
    else:
        movie.tag = "coming"

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

# 9. 正在热映（只展示近 90 天内上映的，过了档期的自动隐藏）
@app.route('/now-playing')
@login_required
def now_playing():
    from datetime import date, timedelta
    today  = date.today().isoformat()
    cutoff = (date.today() - timedelta(days=90)).isoformat()  # 最长院线窗口约 90 天
    movies = Movie.query.filter_by(movie_type='now_playing') \
        .filter(Movie.release_date >= cutoff) \
        .filter(Movie.release_date <= today) \
        .order_by(Movie.rating.desc()).all()
    return render_template('now_playing.html', movies=movies)


# 10. 即将上映（只展示今天及以后上映的）
@app.route('/upcoming')
@login_required
def upcoming():
    from datetime import date
    today = date.today().isoformat()  # 'YYYY-MM-DD' 字符串比较对 ISO 格式天然有效
    movies = Movie.query.filter_by(movie_type='upcoming') \
        .filter(Movie.release_date >= today) \
        .order_by(Movie.release_date.asc()).all()
    return render_template('upcoming.html', movies=movies)


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


# ------------------------------
# ✅ 新增：个人中心 - 我的选座
# ------------------------------
@app.route('/my-seats')
@login_required
def my_seats():
    # ✅ 不使用任何新增字段！只根据 选座时间 模糊匹配当前用户的记录
    my_bookings = Seat.query.filter_by(is_sold=True).all()

    result = []
    for s in my_bookings:
        m = Movie.query.get(s.movie_id)
        result.append({
            "movie_title": m.title if m else "未知电影",
            "movie_poster": m.poster_url if m else "",
            "hall": s.hall,
            "show_time": s.show_time,
            "seat": f"{s.row}排{s.col}座",
            "seat_id": s.id
        })

    return render_template("my_seats.html", bookings=result)

@app.route('/cancel-seat/<int:seat_id>', methods=['POST'])
@login_required
def cancel_seat(seat_id):
    s = Seat.query.get_or_404(seat_id)
    s.is_sold = False
    db.session.commit()
    flash("取消选座成功", "success")
    return redirect(url_for("my_seats"))

# 9. 选座购票功能
@app.route('/ticket-booking')
@login_required
def ticket_booking():
    all_movies = Movie.query.order_by(Movie.id.desc()).all()
    now_showing = all_movies[:8]
    coming_soon = all_movies[8:16]
    
    import random
    for movie in now_showing:
        movie.rating_virtual = round(random.uniform(7.0, 9.5), 1)
    
    from datetime import datetime, timedelta
    for movie in coming_soon:
        days_later = random.randint(1, 30)
        release_date = datetime.now() + timedelta(days=days_later)
        movie.release_date_virtual = release_date.strftime("%Y-%m-%d")
    
    return render_template(
        'ticket_booking.html',
        now_showing=now_showing,
        coming_soon=coming_soon
    )

# 10. 选座页面
@app.route('/seat-selection/<int:movie_id>')
@login_required
def seat_selection(movie_id):
    movie = Movie.query.get_or_404(movie_id)

    # 根据数据库中的 movie_type 字段设置标签
    if movie.movie_type == 'now_playing':
        movie.tag = 'now'
    elif movie.movie_type == 'upcoming':
        movie.tag = 'coming'
    else:
        # 兼容旧数据：如果 movie_type 为空，尝试从 release_date 推断
        if movie.release_date:
            from datetime import date
            today = date.today().isoformat()
            movie.tag = 'coming' if movie.release_date >= today else 'now'
        else:
            # 默认按热映处理（或可根据实际情况调整）
            movie.tag = 'now'

    # 只查询座位
    sold_seats = Seat.query.filter_by(
        movie_id=movie_id,
        hall="1号厅",
        show_time="10:00",
        is_sold=True
    ).all()

    return render_template(
        'seat_selection.html',
        movie=movie,
        sold_seats=sold_seats
    )
# ------------------------------
# ✅ 新增：确认选座（存入数据库）
# ------------------------------
@app.route('/confirm-seats/<int:movie_id>', methods=['POST'])
@login_required
def confirm_seats(movie_id):
    seats_str = request.form.get('seats', '')
    if not seats_str:
        flash('请先选择座位！', 'warning')
        return redirect(url_for('seat_selection', movie_id=movie_id))
    
    seat_list = seats_str.split(',')
    for seat in seat_list:
        row, col = seat.split('-')
        row = int(row)
        col = int(col)
        
        s = Seat.query.filter_by(
            movie_id=movie_id,
            row=row,
            col=col,
            hall="1号厅",
            show_time="10:00"
        ).first()
        
        if not s:
            s = Seat(
                movie_id=movie_id,
                row=row,
                col=col,
                hall="1号厅",
                show_time="10:00",
                is_sold=True,
            )
            db.session.add(s)
        else:
            s.is_sold = True
    
    db.session.commit()
    flash('选座成功！', 'success')
    return redirect(url_for('my_seats'))
# ------------------------------
# 智能座位推荐 API
# ------------------------------
@app.route('/api/recommend_seats/<int:movie_id>', methods=['POST'])
@login_required
def recommend_seats(movie_id):
    """
    接收用户自然语言输入，返回推荐的座位列表
    请求体 JSON: {"text": "两人连坐视野好", "selected": ["1-5","1-6"]}
    返回 JSON: {"recommendations": ["5排5座","5排6座"]}
    """
    data = request.get_json()
    user_text = data.get('text', '').strip()
    selected_seats = data.get('selected', [])  # 格式 ["row-col", ...]

    # 获取已售座位
    sold_seats = Seat.query.filter_by(
        movie_id=movie_id,
        hall="1号厅",
        show_time="10:00",
        is_sold=True
    ).all()
    sold_set = {(s.row, s.col) for s in sold_seats}

    # 合并用户当前已选座位（未确认）
    selected_set = set()
    for seat_str in selected_seats:
        if '-' in seat_str:
            r, c = seat_str.split('-')
            selected_set.add((int(r), int(c)))

    occupied = sold_set | selected_set

    # 解析用户输入
    # 人数
    num_match = re.search(r'(\d+)\s*人|(两|三|四)人|(一|两|三|四)个|(一|两|三|四)人', user_text)
    if num_match:
        if num_match.group(1):
            num = int(num_match.group(1))
        elif num_match.group(2) == '两':
            num = 2
        elif num_match.group(2) == '三':
            num = 3
        elif num_match.group(2) == '四':
            num = 4
        else:
            num = 1
    else:
        num = 2  # 默认两人

    need_adjacent = bool(re.search(r'连坐|一起|挨着|相邻|并排', user_text))
    prefer_center = bool(re.search(r'中间|中央|中心|视野好|好视野', user_text))
    prefer_edge = bool(re.search(r'边上|角落|靠边', user_text))
    prefer_back = bool(re.search(r'后排|靠后', user_text))
    prefer_front = bool(re.search(r'前排|靠前', user_text))

    # 情感分析（关键词）
    is_introvert = bool(re.search(r'内向|安静|独自|不喜欢人|一个人', user_text))
    is_couple = bool(re.search(r'情侣|浪漫|约会|恋爱', user_text))
    if is_introvert:
        num = 1
        need_adjacent = False
    # 辅助函数：计算周围空座位数量
    def count_empty_around(row, col):
        cnt = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 1 <= nr <= 10 and 1 <= nc <= 15 and (nr, nc) not in occupied:
                    cnt += 1
        return cnt

    # 为每个可用座位打分
    candidates = []  # (row, col, score)
    for row in range(1, 11):
        for col in range(1, 16):
            if (row, col) in occupied:
                continue
            score = 0
            # 位置偏好
            if prefer_center and 7 <= col <= 9:
                score += 5
            if prefer_edge and (col <= 3 or col >= 13):
                score += 3
            if prefer_back and row >= 8:
                score += 4
            if prefer_front and row <= 3:
                score += 4
            # 情感偏好
            if is_introvert:
                empty_around = count_empty_around(row, col)
                score += empty_around * 2
            if is_couple:
                if row >= 8 and (col <= 3 or col >= 13):
                    score += 6
                empty_around = count_empty_around(row, col)
                score += empty_around * 1.5
            # 通用：越居中（行）得分越高
            center_row_bonus = 5 - abs(row - 5.5)
            score += center_row_bonus
            candidates.append((row, col, score))

    candidates.sort(key=lambda x: x[2], reverse=True)

    # 如果需要连坐，寻找连续座位组
    def find_adjacent_group(num):
        best_group = None
        best_score = -1
        for row in range(1, 11):
            for start_col in range(1, 16 - num + 1):
                group = [(row, start_col + i) for i in range(num)]
                if all((r, c) not in occupied for r, c in group):
                    # 计算该组平均分
                    total = sum(next(score for r2, c2, score in candidates if r2 == r and c2 == c)
                                for r, c in group)
                    avg = total / num
                    if avg > best_score:
                        best_score = avg
                        best_group = group
        return best_group

    if need_adjacent:
        group = find_adjacent_group(num)
        if group:
            recommendations = [f"{r}排{c}座" for r, c in group]
        else:
            # 找不到连坐，返回高评分的单个座位（但提示无法连坐）
            recommendations = [f"{r}排{c}座" for r, c, _ in candidates[:num]]
    else:
        recommendations = [f"{r}排{c}座" for r, c, _ in candidates[:num]]

    return jsonify({'recommendations': recommendations})

@app.cli.command("upgrade-seats")
def upgrade_seats():
    """为座位表添加价格字段并初始化所有座位价格（幂等）"""
    import sqlite3
    db_path = os.path.join(os.path.dirname(__file__), 'instance', 'movies.db')
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # 检查 price 字段是否存在
    try:
        cur.execute("SELECT price FROM seat LIMIT 1")
        print("✅ price 字段已存在")
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE seat ADD COLUMN price INTEGER DEFAULT 25")
            conn.commit()
            print("✅ 已添加 price 字段")
        except Exception as e:
            print(f"❌ 添加字段失败: {e}")
            conn.close()
            return
    
    # 更新所有座位价格（基于位置，不是基于现有数据）
    # 优选区：第3-8排，第5-11列 = 35元
    # 其他区域 = 25元
    try:
        # 先全部设为25元
        cur.execute("UPDATE seat SET price = 25")
        conn.commit()
        
        # 再更新优选区为35元
        cur.execute("""
            UPDATE seat 
            SET price = 35 
            WHERE (row BETWEEN 3 AND 8) AND (col BETWEEN 5 AND 11)
        """)
        conn.commit()
        premium_count = cur.rowcount
        
        # 统计
        cur.execute("SELECT COUNT(*) FROM seat WHERE price = 35")
        total_premium = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM seat")
        total_seats = cur.fetchone()[0]
        
        print(f"✅ 已更新 {premium_count} 个优选座位为35元")
        print(f"📊 总计：{total_premium} 个优选区(¥35)，{total_seats - total_premium} 个普通区(¥25)")
        
    except Exception as e:
        print(f"⚠️ 更新价格时出错: {e}")
    
    conn.close()
    print("数据库升级完成！")


@app.cli.command("init-seats-for-movie")
@click.argument('movie_id')
def init_seats_for_movie(movie_id):
    """为指定电影初始化所有座位（10排x15列）"""
    from datetime import datetime
    
    movie = Movie.query.get(movie_id)
    if not movie:
        print(f"❌ 电影 {movie_id} 不存在")
        return
    
    # 检查是否已有座位
    existing_count = Seat.query.filter_by(movie_id=movie_id, hall="1号厅", show_time="10:00").count()
    if existing_count > 0:
        print(f"⚠️ 电影 {movie.title} 已有 {existing_count} 个座位，跳过初始化")
        return
    
    # 创建 10排 x 15列 的座位
    created = 0
    for row in range(1, 11):
        for col in range(1, 16):
            # 判断是否为优选区
            is_premium = (3 <= row <= 8 and 5 <= col <= 11)
            price = 35 if is_premium else 25
            
            seat = Seat(
                movie_id=movie_id,
                row=row,
                col=col,
                hall="1号厅",
                show_time="10:00",
                is_sold=False,
                price=price
            )
            db.session.add(seat)
            created += 1
    
    db.session.commit()
    print(f"✅ 已为《{movie.title}》创建 {created} 个座位")
    print(f"   优选区(¥35)：第3-8排，第5-11列")
    print(f"   普通区(¥25)：其他区域")

# ==================== 支付API（弹窗版）====================

@app.route('/api/pay-seats/<int:movie_id>', methods=['POST'])
@login_required
def api_pay_seats(movie_id):
    """处理弹窗支付（用户点击"已完成支付"后）"""
    data = request.get_json()
    seats_data = data.get('seats', [])
    total = data.get('total', 0)
    
    if not seats_data:
        return jsonify({'success': False, 'message': '未选择座位'})
    
    try:
        for seat_info in seats_data:
            row = seat_info['row']
            col = seat_info['col']
            price = seat_info.get('price', 25)
            
            # 查找或创建座位记录
            s = Seat.query.filter_by(
                movie_id=movie_id,
                row=row,
                col=col,
                hall="1号厅",
                show_time="10:00"
            ).first()
            
            if not s:
                s = Seat(
                    movie_id=movie_id,
                    row=row,
                    col=col,
                    hall="1号厅",
                    show_time="10:00",
                    price=price
                )
                db.session.add(s)
            
            # 检查是否已被其他人购买
            if s.is_sold:
                return jsonify({
                    'success': False, 
                    'message': f'{row}排{col}座已被他人购买，请重新选座'
                })
            
            s.is_sold = True
            s.price = price
        
        db.session.commit()
        return jsonify({
            'success': True, 
            'message': f'支付成功，共支付¥{total}',
            'redirect': url_for('my_seats')
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

# ------------------------------
# ✅ 新增：座位模型（不影响原有表）
# ------------------------------
class Seat(db.Model):
    __tablename__ = 'seat'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    row = db.Column(db.Integer, nullable=False)
    col = db.Column(db.Integer, nullable=False)
    hall = db.Column(db.String(20), default="1号厅")
    show_time = db.Column(db.String(20), default="10:00")
    is_sold = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


if __name__ == '__main__':
    app.run(debug=True, port=5000)