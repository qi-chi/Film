"""
内容推荐引擎
支持三种模式：
  1. 冷启动（无评分历史）：返回 TMDB 高分热门电影
  2. 基于内容（有高分记录）：用 BERT 或 TF-IDF 提取电影语义向量，
     构建用户偏好向量后计算余弦相似度，推荐 Top-K
  3. 混合推荐（评分较多时）：融合内容相似度 + NCF 预测分

RAG 检索：
  - TF-IDF 索引会保存 vectorizer，可对用户问题做同空间检索
  - sentence-transformers 索引会保存模型名，查询时懒加载同一模型编码
  - 旧版仅含 movie_ids/embeddings 的 pkl 会走关键词检索兜底
"""

import os
import pickle
import re
import numpy as np

EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), 'models', 'movie_embeddings.pkl')
ST_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# 进程内缓存，避免每次请求都重新加载
_cache = {'movie_ids': None, 'embeddings': None, 'meta': None}

# sentence-transformers 懒加载
_st_model = {'name': None, 'model': None}


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _cosine_similarity(query_vec, matrix):
    """
    query_vec: (1, dim)
    matrix:    (N, dim)
    返回:       (N,) 相似度得分
    """
    q = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-9)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return (q @ m.T)[0]


# ─────────────────────────────────────────────
# 向量计算与持久化
# ─────────────────────────────────────────────

def compute_and_save_embeddings(movies):
    """
    对所有电影计算内容向量并保存。
    优先使用 sentence-transformers (BERT)，若未安装则降级到 TF-IDF。
    可通过 `flask build-embeddings` CLI 命令触发。
    """
    texts, movie_ids = [], []
    for m in movies:
        desc = (m.description or '').strip()[:400]
        text = f"{m.title} {m.genre or ''} {desc}"
        texts.append(text)
        movie_ids.append(m.id)

    embeddings = None

    # ── 尝试 BERT ──
    vectorizer = None
    backend = None
    st_model_name = None

    try:
        from sentence_transformers import SentenceTransformer
        print(f"正在使用 BERT ({ST_MODEL_NAME}) 计算语义向量...")
        model = SentenceTransformer(ST_MODEL_NAME)
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True,
                                  convert_to_numpy=True).astype(np.float32)
        print(f"BERT 编码完成，向量维度: {embeddings.shape}")
        backend = 'st'
        st_model_name = ST_MODEL_NAME
    except ImportError:
        pass

    # ── 降级：TF-IDF ──
    if embeddings is None:
        print("sentence-transformers 未安装，降级使用 TF-IDF 内容向量...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=512, sublinear_tf=True)
        embeddings = vectorizer.fit_transform(texts).toarray().astype(np.float32)
        print(f"TF-IDF 编码完成，向量维度: {embeddings.shape}")
        backend = 'tfidf'

    bundle = {
        'movie_ids': movie_ids,
        'embeddings': embeddings,
        'backend': backend or 'tfidf',
        'vectorizer': vectorizer,
        'st_model_name': st_model_name,
    }

    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(bundle, f)

    _cache['movie_ids'] = movie_ids
    _cache['embeddings'] = embeddings
    _cache['meta'] = {
        'backend': bundle['backend'],
        'vectorizer': vectorizer,
        'st_model_name': st_model_name,
    }
    print(f"已保存 {len(movie_ids)} 部电影的内容向量 → {EMBEDDINGS_PATH}")
    return movie_ids, embeddings


def load_embeddings():
    """从磁盘或内存缓存加载向量，返回 (movie_ids, embeddings) 或 (None, None)。"""
    if _cache['movie_ids'] is not None:
        return _cache['movie_ids'], _cache['embeddings']

    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or 'movie_ids' not in data:
            return None, None
        _cache['movie_ids'] = data['movie_ids']
        _cache['embeddings'] = data['embeddings']
        _cache['meta'] = {
            'backend': data.get('backend', 'legacy'),
            'vectorizer': data.get('vectorizer'),
            'st_model_name': data.get('st_model_name'),
        }
        return _cache['movie_ids'], _cache['embeddings']

    return None, None


def _get_st_model(model_name):
    if _st_model['name'] != model_name or _st_model['model'] is None:
        from sentence_transformers import SentenceTransformer
        _st_model['model'] = SentenceTransformer(model_name)
        _st_model['name'] = model_name
    return _st_model['model']


def _retrieve_rag_movie_ids(query: str, k: int = 8):
    """根据用户问题检索最相关的电影 id 列表（用于 RAG）。"""
    movie_ids, embeddings = load_embeddings()
    if movie_ids is None or embeddings is None:
        return []
    meta = _cache.get('meta') or {}
    backend = meta.get('backend', 'legacy')
    q = (query or '').strip()
    if not q:
        return []

    # TF-IDF：与建库时同一 vectorizer
    if backend == 'tfidf' and meta.get('vectorizer') is not None:
        vec = meta['vectorizer']
        qv = vec.transform([q]).toarray().astype(np.float32)
        sims = _cosine_similarity(qv, embeddings)
        top_idx = np.argsort(-sims)[:k]
        return [movie_ids[int(i)] for i in top_idx]

    # sentence-transformers
    if backend == 'st' and meta.get('st_model_name'):
        try:
            model = _get_st_model(meta['st_model_name'])
            qv = model.encode([q], convert_to_numpy=True).astype(np.float32)
            sims = _cosine_similarity(qv.reshape(1, -1), embeddings)
            top_idx = np.argsort(-sims)[:k]
            return [movie_ids[int(i)] for i in top_idx]
        except Exception:
            pass

    # 旧 pkl 或无可用编码器：关键词兜底
    return _keyword_retrieve_movie_ids(q, k)


def _keyword_retrieve_movie_ids(query: str, k: int):
    from sqlalchemy import or_
    from models import Movie

    tokens = [t for t in re.split(r'[\s,，。.!？；;、]+', query) if len(t) >= 2][:6]
    if not tokens:
        rows = Movie.query.order_by(Movie.rating.desc()).limit(k).all()
        return [m.id for m in rows]

    conds = [
        Movie.title.ilike(f'%{t}%') | Movie.description.ilike(f'%{t}%')
        for t in tokens
    ]
    movies = Movie.query.filter(or_(*conds)).order_by(Movie.rating.desc()).limit(k).all()
    ids = [m.id for m in movies]
    if len(ids) < k:
        seen = set(ids)
        for m in Movie.query.order_by(Movie.rating.desc()).limit(k * 2).all():
            if m.id not in seen:
                ids.append(m.id)
                seen.add(m.id)
            if len(ids) >= k:
                break
    return ids[:k]


def build_rag_context_for_query(query: str, k: int = 6) -> str:
    """
    为 LLM 拼装「检索增强」用的纯文本片段（来自站内 Movie 表）。
    无向量索引时仍可用关键词检索生成片段。
    """
    ids = _retrieve_rag_movie_ids(query, k=max(k, 4))
    if not ids:
        return ""

    from models import Movie

    movies = Movie.query.filter(Movie.id.in_(ids)).all()
    order = {mid: i for i, mid in enumerate(ids)}
    movies.sort(key=lambda m: order.get(m.id, 999))

    lines = []
    for m in movies:
        desc = ((m.description or '').replace('\n', ' '))[:320]
        lines.append(
            f"《{m.title}》 类型 {m.genre or '未知'} "
            f"评分约 {m.rating or 0:.1f} 简介摘录：{desc}"
        )
    return "\n".join(lines)


def embeddings_ready():
    """检查是否已有可用的内容向量缓存。"""
    return os.path.exists(EMBEDDINGS_PATH)


# ─────────────────────────────────────────────
# 三类推荐逻辑
# ─────────────────────────────────────────────

def get_cold_start_recommendations(all_movies, n=10, exclude_ids=None):
    """
    冷启动推荐：按 TMDB 评分降序排列，返回热门高分影片。
    置信度用归一化的 TMDB 评分代替（0.0~1.0）。
    """
    exclude_ids = set(exclude_ids or [])
    candidates = [m for m in all_movies if m.id not in exclude_ids]
    candidates.sort(key=lambda m: (m.rating or 0.0), reverse=True)
    return [(m, float(m.rating or 0.0) / 10.0) for m in candidates[:n]]


def get_content_based_recommendations(liked_ids, all_movies, n=10, exclude_ids=None):
    """
    内容推荐：
      1. 加载全量电影语义向量
      2. 对用户高分电影向量取平均，构建用户偏好向量
      3. 计算偏好向量与所有电影向量的余弦相似度
      4. 按相似度降序返回 Top-N（排除已看和待排除 ID）
    返回 list[(movie, similarity_score)] 或 None（向量未就绪时）
    """
    exclude_ids = set(exclude_ids or [])
    movie_ids, embeddings = load_embeddings()
    if movie_ids is None:
        return None

    id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    liked_indices = [id_to_idx[mid] for mid in liked_ids if mid in id_to_idx]
    if not liked_indices:
        return None

    liked_embs = embeddings[liked_indices]             # (k, dim)
    user_profile = liked_embs.mean(axis=0, keepdims=True)  # (1, dim)

    sims = _cosine_similarity(user_profile, embeddings)    # (N,)

    liked_set = set(liked_ids)
    id_to_movie = {m.id: m for m in all_movies}
    results = []
    for i, mid in enumerate(movie_ids):
        if mid in exclude_ids or mid in liked_set:
            continue
        if mid in id_to_movie:
            results.append((id_to_movie[mid], float(sims[i])))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n]


def get_hybrid_recommendations(ratings, all_movies, ncf_scorer=None, n=10):
    """
    混合推荐主入口：
      - 无评分记录  → 冷启动
      - 有高分记录  → 内容推荐（向量未就绪时自动降级到冷启动）
      - 评分 ≥ 5 条 → 额外融合 NCF 预测分
    
    参数：
      ratings    : Rating 对象列表（已从数据库查出）
      all_movies : Movie 对象列表
      ncf_scorer : callable(movies) → list[float]，NCF 预测分 0~10
      n          : 最少返回数量
    返回：
      list[(movie, confidence_score)]，按置信度降序
      以及 rec_type 字符串 ('cold' | 'content' | 'hybrid')
    """
    watched_ids = {r.movie_id for r in ratings}
    # 高分标准：≥ 7 分（满分 10）
    high_rated_ids = {r.movie_id for r in ratings if r.score >= 7.0}

    candidates_pool = [m for m in all_movies if m.id not in watched_ids]

    # ── 冷启动 ──
    if not high_rated_ids:
        recs = get_cold_start_recommendations(candidates_pool, n=n)
        return recs, 'cold'

    # ── 内容推荐 ──
    request_n = max(n * 2, 20)  # 多取一些，后续 NCF 再筛选
    content_recs = get_content_based_recommendations(
        list(high_rated_ids), all_movies, n=request_n, exclude_ids=watched_ids
    )

    if content_recs is None:
        # 向量未就绪，降级到冷启动
        recs = get_cold_start_recommendations(candidates_pool, n=n)
        return recs, 'cold'

    # ── 混合融合 NCF ──
    if ncf_scorer and len(ratings) >= 5:
        rec_movies = [m for m, _ in content_recs]
        try:
            ncf_scores = ncf_scorer(rec_movies)
            # 随评分数量增加，NCF 权重最高到 0.60
            alpha = min(0.60, 0.10 + len(ratings) * 0.015)
            blended = []
            for (movie, c_sim), ncf_s in zip(content_recs, ncf_scores):
                score = (1 - alpha) * c_sim + alpha * (ncf_s / 10.0)
                blended.append((movie, score))
            blended.sort(key=lambda x: x[1], reverse=True)
            return blended[:n], 'hybrid'
        except Exception:
            pass  # NCF 推断失败时退回纯内容推荐

    return content_recs[:n], 'content'
