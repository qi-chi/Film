# AI 电影推荐系统

基于 Python (Flask) 和 深度学习 (PyTorch) 的电影推荐系统，集成了用户认证、电影展示、数据大屏、个性化推荐以及智能 AI 小助手。

## 功能特性
1. **用户注册与登录**：安全的账号密码管理（Werkzeug hash），首次访问需注册。
2. **电影数据展示**：支持关键词搜索、分页浏览，直观展示电影海报、类型与评分。
3. **电影详情页**：展示电影详细信息，支持用户发表评论和添加收藏。
4. **收藏列表**：专属个人收藏夹，可随时移除不感兴趣的电影。
5. **数据可视化**：基于 ECharts 实现多维度数据大屏（年度发布趋势、流派占比、评分分布）。
6. **智能电影推荐**：内置基于 PyTorch 的协同过滤深度学习推荐模型骨架。
7. **个人中心**：用户可修改邮箱、年龄、性别等个人信息。
8. **AI 智能小助手**：全局悬浮式聊天弹窗，随时为用户提供引导与解答。

## 技术栈
- **后端**：Python 3, Flask, Flask-SQLAlchemy, Flask-Login
- **前端**：HTML5, CSS3, JavaScript, Bootstrap 5, ECharts
- **深度学习**：PyTorch, NumPy, Pandas
- **数据库**：SQLite (开发环境默认)

## 快速启动

1. **安装依赖**
确保你已经安装了 Python 3.8+，然后在项目根目录下运行：
```bash
pip install -r requirements.txt
```

2. **初始化数据库及测试数据**
```bash
flask init-db
```

3. **运行服务**
```bash
python app.py
```
或者
```bash
flask run
```

4. **访问系统**
在浏览器中打开：[http://127.0.0.1:5000](http://127.0.0.1:5000)

## 项目结构
- `app.py`: Flask 主程序及所有路由控制器
- `models.py`: 数据库模型定义 (User, Movie, Favorite, Comment)
- `recommender.py`: 基于 PyTorch 的深度学习推荐算法骨架
- `requirements.txt`: 项目依赖包列表
- `templates/`: HTML 页面模板
- `static/`: 静态资源文件 (CSS/JS/Images)
