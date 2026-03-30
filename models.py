from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    age = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    avatar = db.Column(db.String(200), nullable=True)  # 新增头像字段
    face_encoding = db.Column(db.Text, nullable=True)  # 人脸特征向量（JSON格式存储）

    favorites = db.relationship('Favorite', backref='user', lazy=True)
    comments = db.relationship('Comment', backref='user', lazy=True)
    ratings = db.relationship('Rating', backref='user', lazy=True) # 新增用户评分关系
    circle_posts = db.relationship('CirclePost', backref='user', lazy=True)
    circle_comments = db.relationship('CircleComment', backref='user', lazy=True)

class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.Integer, nullable=True, index=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    release_year = db.Column(db.Integer, nullable=True)
    release_date = db.Column(db.String(20), nullable=True)   # 精确日期，如 2025-06-15
    genre = db.Column(db.String(100), nullable=True)
    rating = db.Column(db.Float, default=0.0)
    poster_url = db.Column(db.String(255), nullable=True)
    # 'popular' | 'now_playing' | 'upcoming'
    movie_type = db.Column(db.String(20), nullable=False, default='popular')
    
    favorites = db.relationship('Favorite', backref='movie', lazy=True)
    comments = db.relationship('Comment', backref='movie', lazy=True)
    ratings = db.relationship('Rating', backref='movie', lazy=True) # 新增电影被评分关系

class Rating(db.Model):
    """新增：用户观影/评分历史记录表"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    score = db.Column(db.Float, nullable=False) # 用户给出的评分
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Circle(db.Model):
    """圈子模型"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    background_color = db.Column(db.String(20), nullable=True, default='#f0f0f0')  # 背景颜色
    background_image = db.Column(db.String(255), nullable=True)  # 背景图片路径
    
    members = db.relationship('CircleMember', backref='circle', lazy=True)
    posts = db.relationship('CirclePost', backref='circle', lazy=True)

class CircleMember(db.Model):
    """圈子成员模型"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    circle_id = db.Column(db.Integer, db.ForeignKey('circle.id'), nullable=False)
    joined_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # 确保用户在一个圈子中只存在一条记录
    __table_args__ = (db.UniqueConstraint('user_id', 'circle_id', name='_user_circle_uc'),)

class CirclePost(db.Model):
    """圈子帖子模型"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    circle_id = db.Column(db.Integer, db.ForeignKey('circle.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    image_url = db.Column(db.String(255), nullable=True)  # 存储图片路径
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    comments = db.relationship('CircleComment', backref='post', lazy=True)

class CircleComment(db.Model):
    """圈子评论模型"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('circle_post.id'), nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('circle_comment.id'), nullable=True)  # 用于回复
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    replies = db.relationship('CircleComment', backref=db.backref('parent', remote_side=[id]), lazy=True)
class Seat(db.Model):
    __tablename__ = 'seat'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    row = db.Column(db.Integer, nullable=False)
    row = db.Column(db.Integer, nullable=False)  # 行号 1-10
    col = db.Column(db.Integer, nullable=False)  # 列号 1-15
    hall = db.Column(db.String(20), default="1号厅")
    show_time = db.Column(db.String(20), default="10:00")
    is_sold = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # ✅ 新增：座位价格（蓝色座位35元，普通座位25元）
    price = db.Column(db.Integer, default=25)
    
    movie = db.relationship('Movie', backref=db.backref('seats', lazy=True))