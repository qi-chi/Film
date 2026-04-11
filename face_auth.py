import base64
import cv2
import json
import numpy as np
import os
from io import BytesIO
from PIL import Image

FEATURE_VERSION = "hybrid_v1"
FEATURE_DIM = 512
DEFAULT_TOLERANCE = 0.52

try:
    from skimage.feature import hog as sk_hog  # type: ignore[reportMissingImports]
    from skimage.feature import local_binary_pattern  # type: ignore[reportMissingImports]

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    sk_hog = None
    local_binary_pattern = None


def get_face_cascade():
    """自动定位 OpenCV haarcascade 模型。"""
    try:
        cv2_dir = os.path.dirname(cv2.__file__)
        path = os.path.join(cv2_dir, "data", "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(path)
        if not cascade.empty():
            return cascade
    except Exception:
        pass
    return None


face_cascade = get_face_cascade()
if face_cascade:
    print("✓ 人脸检测器加载成功")
else:
    print("⚠️  人脸检测已启用兼容模式")

if not SKIMAGE_AVAILABLE:
    print("⚠️  未安装 scikit-image，LBP/HOG 将自动降级")


def _preprocess_gray_variants(gray):
    """构造多种灰度预处理版本，提高弱光/低对比度下的检测率。"""
    variants = [gray, cv2.equalizeHist(gray)]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(clahe.apply(gray))
    return variants


def _detect_faces_multi_pass(gray):
    """
    多阶段人脸检测：
    - 多种预处理（原图/均衡化/CLAHE）
    - 多组 detectMultiScale 参数
    """
    if face_cascade is None:
        return []

    params = [
        {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (64, 64)},
        {"scaleFactor": 1.08, "minNeighbors": 4, "minSize": (48, 48)},
        {"scaleFactor": 1.05, "minNeighbors": 3, "minSize": (40, 40)},
    ]

    all_faces = []
    for variant in _preprocess_gray_variants(gray):
        for p in params:
            faces = face_cascade.detectMultiScale(
                variant,
                scaleFactor=p["scaleFactor"],
                minNeighbors=p["minNeighbors"],
                minSize=p["minSize"],
            )
            for f in faces:
                all_faces.append(tuple(int(v) for v in f))

    return list(dict.fromkeys(all_faces))


def _pick_largest_face(faces):
    """多人脸时选面积最大的人脸，避免误检导致直接失败。"""
    if not faces:
        return None
    return max(faces, key=lambda item: item[2] * item[3])


def _resample_vector(vec, target_len):
    """将任意长度向量插值到固定长度。"""
    vec = np.asarray(vec, dtype=np.float32).ravel()
    if vec.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if vec.size == target_len:
        return vec
    src_idx = np.linspace(0.0, 1.0, num=vec.size, dtype=np.float32)
    dst_idx = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(dst_idx, src_idx, vec).astype(np.float32)


def _extract_lbp_features(image, num_points=24, radius=3):
    """提取 LBP 纹理特征。"""
    if not SKIMAGE_AVAILABLE:
        return np.array([], dtype=np.float32)
    lbp = local_binary_pattern(image, num_points, radius, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, num_points + 3),
        range=(0, num_points + 2),
    )
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)
    return hist


def _extract_hog_features(image):
    """提取 HOG 形状特征。"""
    if not SKIMAGE_AVAILABLE:
        return np.array([], dtype=np.float32)
    features = sk_hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True,
    )
    return np.asarray(features, dtype=np.float32)


def _preprocess_face(face_img, target_size=(128, 128)):
    """预处理人脸图像：resize + 直方图均衡 + 轻微降噪。"""
    face_resized = cv2.resize(face_img, target_size)
    face_eq = cv2.equalizeHist(face_resized)
    face_blur = cv2.GaussianBlur(face_eq, (3, 3), 0)
    return face_blur


def _build_hybrid_features(face_processed):
    """
    构建固定 512 维特征：
    - 像素（224）
    - LBP（32）
    - HOG（256）
    合计 512 维。
    """
    pixel_raw = (face_processed.flatten() / 255.0)[::4]
    lbp_raw = _extract_lbp_features(face_processed)
    hog_raw = _extract_hog_features(face_processed)

    pixel_part = _resample_vector(pixel_raw, 224)
    lbp_part = _resample_vector(lbp_raw, 32)
    hog_part = _resample_vector(hog_raw, 256)

    features = np.concatenate([pixel_part, lbp_part, hog_part]).astype(np.float32)

    # 标准化 + L2 归一化，便于后续相似度比较
    std = float(np.std(features))
    if std > 1e-8:
        features = (features - np.mean(features)) / std
    norm = float(np.linalg.norm(features))
    if norm > 1e-8:
        features = features / norm

    return features[:FEATURE_DIM]


def detect_face(img):
    """仅检测是否存在人脸。"""
    if face_cascade is None:
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _detect_faces_multi_pass(gray)
    return len(faces) > 0


def extract_face_encoding(image_data):
    """从 base64 图片提取固定维度人脸特征。"""
    try:
        if "," in image_data:
            _, image_data = image_data.split(",", 1)
        image_bytes = base64.b64decode(image_data)

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if face_cascade is not None:
            faces = _detect_faces_multi_pass(gray)
            if len(faces) == 0:
                return False, "未检测到人脸，请确保面部在摄像头范围内"

            x, y, w, h = _pick_largest_face(faces)
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.2)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(gray.shape[1], x + w + pad_x)
            y2 = min(gray.shape[0], y + h + pad_y)
            face_roi = gray[y1:y2, x1:x2]
        else:
            # 兼容模式：取中心正方形区域
            h, w = gray.shape
            side = min(w, h)
            x = (w - side) // 2
            y = (h - side) // 2
            face_roi = gray[y : y + side, x : x + side]

        if face_roi.size == 0:
            return False, "人脸区域提取失败，请重试"

        face_processed = _preprocess_face(face_roi, (128, 128))
        features = _build_hybrid_features(face_processed)
        return True, features.tolist()
    except Exception as e:
        return False, f"人脸特征提取失败: {str(e)}"


def compare_faces(known_encoding, unknown_encoding, tolerance=DEFAULT_TOLERANCE):
    """
    使用余弦距离进行比对（越小越相似）。
    兼容历史旧编码（长度不一致时自动裁剪到最短长度）。
    """
    try:
        if isinstance(known_encoding, str):
            known_encoding = json.loads(known_encoding)
        if isinstance(unknown_encoding, str):
            unknown_encoding = json.loads(unknown_encoding)

        known = np.asarray(known_encoding, dtype=np.float32).ravel()
        unknown = np.asarray(unknown_encoding, dtype=np.float32).ravel()

        if known.size == 0 or unknown.size == 0:
            return False, float("inf")

        if known.size != unknown.size:
            min_len = min(known.size, unknown.size)
            known = known[:min_len]
            unknown = unknown[:min_len]

        known_norm = float(np.linalg.norm(known))
        unknown_norm = float(np.linalg.norm(unknown))
        if known_norm <= 1e-8 or unknown_norm <= 1e-8:
            return False, float("inf")

        known = known / known_norm
        unknown = unknown / unknown_norm

        cosine_similarity = float(np.clip(np.dot(known, unknown), -1.0, 1.0))
        distance = 1.0 - cosine_similarity
        return distance <= tolerance, float(distance)
    except Exception:
        return False, float("inf")


def find_best_match(unknown_encoding, user_encodings, tolerance=DEFAULT_TOLERANCE):
    """在多个用户中找出最佳匹配。"""
    best_match = None
    best_distance = float("inf")
    debug_info = {"feature_version": FEATURE_VERSION, "metric": "cosine_distance"}

    for user in user_encodings:
        if not user.get("face_encoding"):
            continue

        is_match, distance = compare_faces(user["face_encoding"], unknown_encoding, tolerance)
        debug_info[user["username"]] = round(distance, 4)

        if is_match and distance < best_distance:
            best_distance = distance
            best_match = user

    # 没有匹配时，稍微放宽一次阈值，减少误拒绝
    if not best_match and user_encodings:
        relaxed_tolerance = min(tolerance + 0.08, 0.68)
        for user in user_encodings:
            if not user.get("face_encoding"):
                continue
            is_match, distance = compare_faces(
                user["face_encoding"], unknown_encoding, relaxed_tolerance
            )
            if is_match and distance < best_distance:
                best_distance = distance
                best_match = user
        debug_info["relaxed_tolerance"] = round(relaxed_tolerance, 3)

    debug_info["best_distance"] = round(best_distance, 4)
    debug_info["tolerance"] = round(tolerance, 3)
    return best_match, best_distance, debug_info


def encoding_to_json(encoding_list):
    """将特征向量转换为 JSON 字符串存储。"""
    return json.dumps(encoding_list)


def json_to_encoding(json_string):
    """将 JSON 字符串转换回特征向量。"""
    return json.loads(json_string)