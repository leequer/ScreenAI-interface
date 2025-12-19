from flask import Flask, request, jsonify
import os
import sys
import uuid
import traceback
import multiprocessing
from os.path import join as pjoin

# 设置多进程启动方法
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 已经设置过

# 禁用PaddleOCR的多进程，避免冲突
os.environ['PADDLE_USE_MULTIPROCESSING'] = '0'

# 添加UIED目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'UIED'))

app = Flask(__name__)
app.config['DEBUG'] = False  # 禁用调试模式，避免多进程问题

# 配置上传文件夹和允许的文件扩展名
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 检查文件扩展名是否允许
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({'message': 'Flask API is working!'}), 200

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # 检查文件名是否为空
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # 检查文件是否允许上传
        if file and allowed_file(file.filename):
            # 生成唯一的文件名
            unique_filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            app.logger.info(f"File saved to {file_path}")
            
            # 调用UIED的截图识别功能
            from run_single import resize_height_by_longest_edge
            import detect_text.text_detection as text
            import detect_compo.ip_region_proposal as ip
            import detect_merge.merge as merge
            from cnn.CNN import CNN
            
            # 设置输出目录
            output_root = 'api_output'
            os.makedirs(output_root, exist_ok=True)
            os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
            os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
            os.makedirs(pjoin(output_root, 'merge'), exist_ok=True)
            
            # 配置参数
            key_params = {'min-grad':10, 'ffl-block':5, 'min-ele-area':50,
                          'merge-contained-ele':True, 'merge-line-to-paragraph':False, 'remove-bar':True}
            
            # 调整图片大小
            resized_height = resize_height_by_longest_edge(file_path, resize_length=800)
            app.logger.info(f"Resized height: {resized_height}")
            
            # 执行OCR
            app.logger.info("Starting OCR detection")
            text.text_detection(file_path, output_root, show=False, method='paddle')
            app.logger.info("OCR detection completed")
            
            # 执行组件检测
            app.logger.info("Starting component detection")
            classifier = {}
            classifier['Elements'] = CNN('Elements')
            ip.compo_detection(file_path, output_root, key_params,
                               classifier=classifier, resize_by_height=resized_height, show=False)
            app.logger.info("Component detection completed")
            
            # 执行合并
            app.logger.info("Starting merge process")
            name = unique_filename.split('.')[0]
            compo_path = pjoin(output_root, 'ip', str(name) + '.json')
            ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
            merge.merge(file_path, compo_path, ocr_path, pjoin(output_root, 'merge'),
                        is_remove_bar=key_params['remove-bar'], is_paragraph=key_params['merge-line-to-paragraph'], show=False)
            app.logger.info("Merge process completed")
            
            # 准备返回结果
            result = {
                'success': True,
                'json_path': f'{output_root}/merge/{name}.json',
                'image_path': f'{output_root}/merge/{name}.png'  # 合并后的图片通常是png格式
            }
            
            return jsonify(result), 200
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    # 设置多进程启动方法（如果需要）
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # 禁用PaddleOCR多进程以避免错误
    os.environ['PADDLE_USE_MULTIPROCESSING'] = '0'
    
    app.run(debug=False, host='127.0.0.1', port=5000)
