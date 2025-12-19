from flask import Flask, request, jsonify, send_from_directory
import os
import tempfile
import shutil
from os.path import join as pjoin
import sys
import json

# 添加UIED目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'UIED'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.getcwd(), 'outputs')

# 创建必要的目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': 'UIED Image Recognition API',
        'endpoints': {
            '/api/recognize': 'POST - Upload an image to recognize UI components'
        }
    })

@app.route('/api/recognize', methods=['POST'])
def recognize_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    # 获取上传的文件
    image_file = request.files['image']
    
    # 生成唯一文件名
    import uuid
    filename = str(uuid.uuid4()) + '_' + image_file.filename
    input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # 保存上传的文件
    image_file.save(input_image_path)
    
    # 创建输出目录
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], os.path.splitext(filename)[0])
    os.makedirs(output_dir, exist_ok=True)
    
    # 调用UIED的识别功能
    result = run_uied_recognition(input_image_path, output_dir)
    
    if 'error' in result:
        return jsonify(result), 500
    
    # 读取JSON结果并返回
    response = {
        'message': 'Recognition completed successfully',
        'input_image': filename,
        'results': {}
    }
    
    # 读取并返回IP JSON
    if os.path.exists(result['output']['ip_json']):
        with open(result['output']['ip_json'], 'r', encoding='utf-8') as f:
            response['results']['ip'] = json.load(f)
    
    # 读取并返回OCR JSON
    if os.path.exists(result['output']['ocr_json']):
        with open(result['output']['ocr_json'], 'r', encoding='utf-8') as f:
            response['results']['ocr'] = json.load(f)
    
    # 读取并返回Merge JSON
    if os.path.exists(result['output']['merge_json']):
        with open(result['output']['merge_json'], 'r', encoding='utf-8') as f:
            response['results']['merge'] = json.load(f)
    
    return jsonify(response), 200

def run_uied_recognition(input_image_path, output_root):
    try:
        import cv2
        import numpy as np
        import multiprocessing
        
        # 设置多进程启动方式
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        os.environ['PADDLE_USE_MULTIPROCESSING'] = '0'
        
        # 定义参数
        key_params = {'min-grad':10, 'ffl-block':5, 'min-ele-area':50,
                      'merge-contained-ele':True, 'merge-line-to-paragraph':False, 'remove-bar':True}
        
        # 调整图片大小
        def resize_height_by_longest_edge(img_path, resize_length=800):
            org = cv2.imread(img_path)
            height, width = org.shape[:2]
            if height > width:
                return resize_length
            else:
                return int(resize_length * (height / width))
        
        resized_height = resize_height_by_longest_edge(input_image_path, resize_length=800)
        
        is_ip = True
        is_clf = True
        is_ocr = True
        is_merge = True
        
        # 执行OCR
        if is_ocr:
            import detect_text.text_detection as text
            os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
            text.text_detection(input_image_path, output_root, show=False, method='paddle')
        
        # 执行组件检测
        if is_ip:
            import detect_compo.ip_region_proposal as ip
            os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
            classifier = None
            if is_clf:
                classifier = {}
                from cnn.CNN import CNN
                classifier['Elements'] = CNN('Elements')
            ip.compo_detection(input_image_path, output_root, key_params,
                               classifier=classifier, resize_by_height=resized_height, show=False)
        
        # 执行合并
        if is_merge:
            import detect_merge.merge as merge
            os.makedirs(pjoin(output_root, 'merge'), exist_ok=True)
            name = os.path.basename(input_image_path).split('.')[0]
            compo_path = pjoin(output_root, 'ip', str(name) + '.json')
            ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
            merge.merge(input_image_path, compo_path, ocr_path, pjoin(output_root, 'merge'),
                        is_remove_bar=key_params['remove-bar'], is_paragraph=key_params['merge-line-to-paragraph'], show=False)
        
        # 构建返回结果
        result = {
            'input_image': input_image_path,
            'output': {
                'ip_json': pjoin(output_root, 'ip', 'input.json'),
                'ocr_json': pjoin(output_root, 'ocr', 'input.json'),
                'merge_json': pjoin(output_root, 'merge', 'input.json')
            }
        }
        
        return result
        
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
