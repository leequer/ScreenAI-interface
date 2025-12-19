import requests
import os

print('Starting API test...')

# API 端点 URL
url = 'http://127.0.0.1:5000/upload'
print(f'Testing API endpoint: {url}')

# 测试图片路径
test_image_path = 'data/input/test1.png'
print(f'Using test image: {test_image_path}')

# 检查图片是否存在
if not os.path.exists(test_image_path):
    print(f'Error: Test image not found at {test_image_path}')
    exit(1)

print(f'Uploading test image: {test_image_path}')

try:
    # 准备文件上传
    with open(test_image_path, 'rb') as f:
        files = {'image': (os.path.basename(test_image_path), f, 'image/png')}
        print('Sending POST request...')
        response = requests.post(url, files=files)
        print(f'Response received with status code: {response.status_code}')
        print(f'Response content: {response.text}')
        
        if response.status_code == 200:
            print('API test successful!')
            result = response.json()
            print(f'Success: {result.get("success")}')
            print(f'Filename: {result.get("filename")}')
            print(f'JSON results:')
            for key, value in result.get('json_results', {}).items():
                print(f'  {key}: {value}')
        else:
            print('API test failed!')
            
except Exception as e:
    print(f'Error during API test: {str(e)}')