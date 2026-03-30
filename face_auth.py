import cv2
import json
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image

def get_face_cascade():
    try:
        cv2_dir = os.path.dirname(cv2.__file__)
        path = os.path.join(cv2_dir, "data", "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(path)
        if not cascade.empty():
            return cascade
    except:
        pass
    return None

face_cascade = get_face_cascade()

if face_cascade:
    print("✓ 人脸检测器加载成功")
else:
    print("⚠️  人脸检测已启用兼容模式")

def extract_lbp_features(image, num_points=24, radius=3):
    """提取 LBP（局部二值模式）特征 - 纹理特征"""
    try:
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(image, num_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist
    except ImportError:
        return np.array([])

def extract_hog_features(image):
    """提取 HOG（方向梯度直方图）特征 - 形状特征"""
    try:
        from skimage.feature import hog
        features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False, feature_vector=True)
        return features
    except ImportError:
        return np.array([])

def preprocess_face(face_img, target_size=(128, 128)):
    """预处理人脸图像"""
    face_resized = cv2.resize(face_img, target_size)
    face_eq = cv2.equalizeHist(face_resized)
    face_blur = cv2.GaussianBlur(face_eq, (3, 3), 0)
    return face_blur

def extract_face_encoding(image_data):
    """
    从 base64 图片中提取人脸特征（高级版）
    使用 LBP + HOG + 像素特征组合，大幅提高准确率
    """
    try:
        header, image_data = image_data.split(',')
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if face_cascade is not None:
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) == 0:
                return False, "未检测到人脸，请确保面部正对摄像头、光线充足"
            if len(faces) > 1:
                return False, "检测到多张人脸，请确保只有一个人"
            
            x, y, w, h = faces[0]
            padding = int(max(w, h) * 0.3)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(gray.shape[1] - x, w + 2 * padding)
            h = min(gray.shape[0] - y, h + 2 * padding)
            face_roi = gray[y:y+h, x:x+w]
        else:
            h, w = gray.shape
            size = min(w, h)
            x = (w - size) // 2
            y = (h - size) // 2
            face_roi = gray[y:y+size, x:x+size]
        
        face_processed = preprocess_face(face_roi, (128, 128))
        
        features = []
        
        pixel_features = face_processed.flatten() / 255.0
        pixel_features = pixel_features[::4]
        features.extend(pixel_features.tolist())
        
        lbp_features = extract_lbp_features(face_processed)
        if len(lbp_features) > 0:
            features.extend(lbp_features.tolist())
        
        hog_features = extract_hog_features(face_processed)
        if len(hog_features) > 0:
            hog_downsampled = hog_features[::2]
            features.extend(hog_downsampled.tolist())
        
        features = np.array(features)
        if np.std(features) > 0:
            features = (features - np.mean(features)) / np.std(features)
        
        features = features.tolist()
        
        if len(features) < 256:
            features.extend([0.0] * (256 - len(features)))
        
        features = features[:512]
        
        return True, features
        
    except Exception as e:
        return False, f"人脸特征提取失败: {str(e)}"

def compare_faces(known_encoding, unknown_encoding, tolerance=0.5):
    """
    比对人脸特征向量（使用余弦相似度）
    tolerance: 越小越严格，建议 0.4-0.6
    """
    try:
        if isinstance(known_encoding, str):
            known_encoding = json.loads(known_encoding)
        if isinstance(unknown_encoding, str):
            unknown_encoding = json.loads(unknown_encoding)
            
        known = np.array(known_encoding)
        unknown = np.array(unknown_encoding)
        
        if len(known) != len(unknown):
            min_len = min(len(known), len(unknown))
            known = known[:min_len]
            unknown = unknown[:min_len]
        
        dot_product = np.dot(known, unknown)
        norm_known = np.linalg.norm(known)
        norm_unknown = np.linalg.norm(unknown)
        
        if norm_known == 0 or norm_unknown == 0:
            return False, float('inf')
        
        cosine_similarity = dot_product / (norm_known * norm_unknown)
        distance = 1 - cosine_similarity
        
        is_match = distance <= tolerance
        
        return is_match, float(distance)
        
    except Exception as e:
        return False, float('inf')

def find_best_match(unknown_encoding, user_encodings, tolerance=0.5):
    """
    在多个用户中找出最佳匹配
    """
    best_match = None
    best_distance = float('inf')
    debug_info = {}
    
    for user in user_encodings:
        if not user.get('face_encoding'):
            continue
            
        is_match, distance = compare_faces(
            user['face_encoding'], 
            unknown_encoding, 
            tolerance
        )
        
        debug_info[user['username']] = round(distance, 4)
        
        if distance < best_distance:
            best_distance = distance
            if is_match:
                best_match = user
    
    debug_info['best_distance'] = round(best_distance, 4)
    return best_match, best_distance, debug_info

def encoding_to_json(encoding_list):
    """将特征向量转换为 JSON 字符串存储"""
    return json.dumps(encoding_list)

def json_to_encoding(json_string):
    """将 JSON 字符串转换回特征向量"""
    return json.loads(json_string)
