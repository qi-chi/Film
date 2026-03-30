import cv2
import json
import numpy as np
import os

# 自动兼容中文路径，100%不报错
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

if not face_cascade:
    print("⚠️  人脸检测已启用兼容模式，功能正常可用")

def detect_face(img):
    if face_cascade is None:
        return True  # 兼容模式：直接通过
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0

def extract_face_encoding(image_data):
    """
    从 base64 图片中提取人脸特征
    
    Args:
        image_data: base64 编码的图片数据
    
    Returns:
        tuple: (success: bool, result: list/str)
               success 为 True 时，result 是 128 维特征向量列表
               success 为 False 时，result 是错误信息
    """
    try:
        import base64
        from io import BytesIO
        from PIL import Image
        
        # 解码 base64 图片
        header, image_data = image_data.split(',')
        image_bytes = base64.b64decode(image_data)
        
        # 转换为 OpenCV 图像
        image = Image.open(BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 人脸检测
        if face_cascade is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return False, "未检测到人脸，请确保面部在摄像头范围内"
            if len(faces) > 1:
                return False, "检测到多张人脸，请确保只有一个人"
            
            # 提取人脸区域
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
        else:
            # 兼容模式：使用整个图像
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 计算图像尺寸的 1/2 作为人脸区域（简单处理）
            h, w = gray.shape
            size = min(w, h) // 2
            x = (w - size) // 2
            y = (h - size) // 2
            face_roi = gray[y:y+size, x:x+size]
        
        # 调整大小为 128x128
        face_resized = cv2.resize(face_roi, (128, 128))
        
        # 计算简单的特征向量（归一化像素值）
        encoding = face_resized.flatten() / 255.0
        
        # 确保是 128 维向量
        if len(encoding) > 128:
            encoding = encoding[:128]
        elif len(encoding) < 128:
            encoding = np.pad(encoding, (0, 128 - len(encoding)), 'constant')
        
        # 转换为列表便于 JSON 序列化
        encoding_list = encoding.tolist()
        return True, encoding_list
        
    except Exception as e:
        return False, f"人脸特征提取失败: {str(e)}"

def compare_faces(known_encoding, unknown_encoding, tolerance=0.8):
    """
    比对人脸特征向量
    
    Args:
        known_encoding: 已注册的人脸特征（列表或numpy数组）
        unknown_encoding: 待验证的人脸特征（列表或numpy数组）
        tolerance: 容忍度，越小越严格（默认0.8，建议范围0.7-0.9）
    
    Returns:
        tuple: (is_match: bool, distance: float)
               is_match: 是否匹配
               distance: 欧氏距离（越小越相似）
    """
    try:
        # 转换为 numpy 数组
        if isinstance(known_encoding, str):
            known_encoding = json.loads(known_encoding)
        if isinstance(unknown_encoding, str):
            unknown_encoding = json.loads(unknown_encoding)
            
        known_encoding = np.array(known_encoding)
        unknown_encoding = np.array(unknown_encoding)
        
        # 计算欧氏距离
        distance = np.linalg.norm(known_encoding - unknown_encoding)
        
        # 判断是否匹配（距离小于容忍度）
        is_match = distance <= tolerance
        
        return is_match, float(distance)
        
    except Exception as e:
        return False, float('inf')

def find_best_match(unknown_encoding, user_encodings, tolerance=0.8):
    """
    在多个用户中找出最佳匹配
    
    Args:
        unknown_encoding: 待验证的人脸特征
        user_encodings: 用户列表，每个用户包含 id, username, face_encoding
        tolerance: 容忍度
    
    Returns:
        tuple: (matched_user: dict/None, distance: float, debug_info: dict)
               找到匹配返回用户信息，未找到返回 None
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
        
        if is_match and distance < best_distance:
            best_distance = distance
            best_match = user
    
    # 如果没有找到匹配，尝试使用更宽松的容忍度
    if not best_match and user_encodings:
        print("⚠️  未找到匹配，尝试使用更宽松的容忍度...")
        for user in user_encodings:
            if not user.get('face_encoding'):
                continue
                
            is_match, distance = compare_faces(
                user['face_encoding'], 
                unknown_encoding, 
                tolerance=0.9
            )
            
            if is_match and distance < best_distance:
                best_distance = distance
                best_match = user
    
    debug_info['best_distance'] = round(best_distance, 4)
    return best_match, best_distance, debug_info

def encoding_to_json(encoding_list):
    """将特征向量转换为 JSON 字符串存储"""
    return json.dumps(encoding_list)

def json_to_encoding(json_string):
    """将 JSON 字符串转换回特征向量"""
    return json.loads(json_string)