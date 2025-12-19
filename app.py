import os
import json
import uuid
import subprocess
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'data/output'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/')
def index():
    return jsonify({'message': 'Flask API is running successfully!'}), 200

# 确保上传和输出目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 检查文件扩展名是否允许
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # 生成唯一文件名
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(upload_path)
        
        try:
            # 使用 subprocess 调用 UIED 的 run_single.py 脚本
            result = subprocess.run([
                'python',
                'UIED/run_single.py',
                upload_path
            ], capture_output=True, text=True, cwd=os.getcwd())
            
            print('run_single.py output:', result.stdout)
            print('run_single.py error:', result.stderr)
            
            if result.returncode != 0:
                return jsonify({'error': f'UIED processing failed: {result.stderr}'}), 500
            
            # 准备返回数据
            name = os.path.basename(upload_path).split('.')[0]
            result_data = {
                'success': True,
                'filename': unique_filename,
                'json_results': {
                    'ip': os.path.join(app.config['OUTPUT_FOLDER'], 'ip', f'{name}.json'),
                    'ocr': os.path.join(app.config['OUTPUT_FOLDER'], 'ocr', f'{name}.json'),
                    'merge': os.path.join(app.config['OUTPUT_FOLDER'], 'merge', f'{name}.json')
                },
                'image_paths': {
                    # 这里可以根据实际情况添加生成的图片路径
                }
            }
            
            return jsonify(result_data), 200
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)