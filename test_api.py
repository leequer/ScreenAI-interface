import requests

# 测试API端点
url = 'http://localhost:5000/api/analyze'

# 上传的文件路径
file_path = 'data/input/test1.png'

print(f"Testing API at: {url}")
print(f"Uploading file: {file_path}")

# 发送POST请求
try:
    with open(file_path, 'rb') as f:
        files = {'file': (file_path.split('/')[-1], f, 'image/png')}
        response = requests.post(url, files=files, timeout=30)
    
    print('Status Code:', response.status_code)
    print('Response Headers:', dict(response.headers))
    print('Response Text:', response.text)
    
    try:
        print('Response JSON:', response.json())
    except Exception as e:
        print('Error parsing JSON:', str(e))
        print('Content type:', response.headers.get('Content-Type'))
        print('Content length:', response.headers.get('Content-Length'))
        
except Exception as e:
    print('Request failed:', str(e))
