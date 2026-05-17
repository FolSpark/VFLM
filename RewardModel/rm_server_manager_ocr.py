import re
import json
import requests
import base64
import random
import time
from collections import Counter
from PIL import Image
from io import BytesIO
import xml.etree.ElementTree as ET
from datetime import datetime
from flask import Flask, request, jsonify
from multiprocessing import Manager

app = Flask(__name__)

# 创建共享队列
manager = Manager()
SERVER_QUEUE = manager.Queue()

ocr_server_urls = [
    # "http://127.0.0.1:30940/predict/ocr_system", 
    # "http://127.0.0.1:30941/predict/ocr_system", 
    # "http://127.0.0.1:30942/predict/ocr_system", 
    # "http://127.0.0.1:30943/predict/ocr_system", 
    # "http://127.0.0.1:30944/predict/ocr_system", 
    # "http://127.0.0.1:30945/predict/ocr_system", 
    "http://210.45.70.152:30946/predict/ocr_system",
    "http://210.45.70.152:30947/predict/ocr_system"
]


def extract_svg_text(svg_code):
    # 解析SVG并清理命名空间
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}
    # tree = ET.parse(svg_path)
    # svg_code = open(svg_path, 'r', encoding='utf-8').read()
    tree = ET.ElementTree(ET.fromstring(svg_code))
    root = tree.getroot()
    
    # 清理所有命名空间前缀
    for elem in root.iter():
        elem.tag = elem.tag.split('}', 1)[1]  # 移除命名空间
    
    # 递归提取文本内容
    def get_text(element):
        text_parts = []
        # 提取当前元素的文本内容
        if element.text and element.text.strip():
            text_parts.append(element.text.strip())
        # 遍历子元素
        # for child in element:
        #     text_parts.extend(get_text(child))
        # 提取尾部文本
        if element.tail and element.tail.strip():
            text_parts.append(element.tail.strip())
        return text_parts
    
    # 收集所有text/tspan元素内容
    all_text = []
    for elem in root.iter():
        if elem.tag in ('text', 'tspan'):
            all_text.extend(get_text(elem))
    
    # 合并文本并处理多余空白
    return ' '.join(' '.join(part.split()) for part in all_text)


def find_invalid_characters(s):
    """去除字符串中非中英文、数字的字符"""
    # 使用正则表达式，保留中英文、数字
    pattern = re.compile(r'[^a-zA-Z0-9\u4e00-\u9fff，。！!？?：；、“”‘’（）【】「」¥￥…—｜《》·～\(\)\{\}\[\]\'\"@#\$%\^&\*\-\+_\/|<>`~]')
    # pattern = re.compile(r'[^a-zA-Z0-9\u4e00-\u9fff]')
    return pattern.sub('', s)


def remove_chars_by_count(s1, s2):
    # s1中删除s2中出现的字符，保留s1中未在s2中出现的字符
    # 统计s2中每个字符的出现次数
    s2_counts = Counter(s2)
    # 初始化结果列表和计数器
    result = []
    s1_counts = Counter()
    
    for char in s1:
        # 如果字符在s2中且未达到删除次数上限
        if char in s2_counts and s1_counts.get(char, 0) < s2_counts[char]:
            s1_counts[char] += 1
        else:
            result.append(char)
    
    return ''.join(result)


def calculate_metrics(pro_words, ocr_words):
    pro_counter = Counter(pro_words)
    ocr_counter = Counter(ocr_words)

    # Calculate TP, FN, FP considering character frequencies
    # TP is the sum of min counts for each character present in both
    tp = sum((pro_counter & ocr_counter).values())  # Intersection of counts

    # FN: For each character in pro_words, the difference between pro and ocr counts
    fn = sum(max(0, pro_counter[char] - ocr_counter.get(char, 0)) for char in pro_counter)

    # FP: For each character in ocr_words, the difference between ocr and pro counts
    fp = sum(max(0, ocr_counter[char] - pro_counter.get(char, 0)) for char in ocr_counter)

    # Calculate metrics
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0

    return recall, precision, accuracy

def get_image_ocr(image_b64_str):
    """
    调用OCR服务，返回图片中的所有文本
    :param image_b64_str: 形如"data:image;base64,..."的字符串
    :return: all_text字符串
    """
    # 解析base64字符串，获取图片数据
    if image_b64_str.startswith("data:image;base64,"):
        encoded_image = image_b64_str[len("data:image;base64,"):]
    else:
        encoded_image = image_b64_str

    # 随机选择一个OCR服务端
    data = {"images": [encoded_image]}
    headers = {"Content-type": "application/json"}

    for _ in range(3):
        try:
            server_url = random.choice(ocr_server_urls)
            response = requests.post(url=server_url, headers=headers, data=json.dumps(data), timeout=60)
            # print(response.json())
            result = response.json()["results"][0]
            all_text = ""
            for detect in result:
                all_text += detect.get("text", "")
            return all_text
        except Exception as e:
            print("OCR服务调用出错: ", e)
            time.sleep(0.1)
    return ""

def get_ocr_metrics(request_data):
    """
    计算OCR文本与参考文本的指标
    :param image_b64_str: 形如"data:image;base64,..."的字符串
    :param pro_text: 参考文本
    :return: recall, precision, accuracy
    """
    pro_text = request_data["prompt"]
    svg_code = request_data["svg_code"]
    bg_img_b64_str = request_data["image1"]
    image_b64_str = request_data["image2"]
    
    bg_ocr_text = get_image_ocr(bg_img_b64_str)
    ocr_text = get_image_ocr(image_b64_str)
    ocr_text = remove_chars_by_count(ocr_text, bg_ocr_text)
    try:
        svg_code_text = extract_svg_text(svg_code)
    except Exception as e:
        print(f"Error extracting SVG text: {e}")
        svg_code_text = ""
    print(f"OCR识别结果: {ocr_text}")
    
    # 清洗文本，去除无效字符
    clean_pro_text = find_invalid_characters(pro_text)
    clean_ocr_text = find_invalid_characters(ocr_text)
    clean_svg_code_text = find_invalid_characters(svg_code_text)
    
    # 计算指标
    recall, precision, accuracy = calculate_metrics(clean_pro_text, clean_ocr_text)
    svg_recall, svg_precision, svg_accuracy = calculate_metrics(clean_pro_text, clean_svg_code_text)
    # return {"ocr_text": ocr_text, "recall": recall, "precision": precision, "accuracy": accuracy}
    return {"ocr_text": ocr_text, "svg_code_text": svg_code_text, "ocr_accuracy": accuracy, "svg_code_accuracy": svg_accuracy}

def get_image_b64_str(image):
    """
    读取图片并转换为base64编码字符串
    :param image: PIL.Image对象
    :return: base64编码的字符串
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image;base64,{encoded_image_text}"

def test_server_connection(server_url):
    """测试与服务端的连接"""
    try:
        headers = {
            "Content-Type": "application/json",
        }
        demo_data = {
            "prompt": "text_content",
            "image1": get_image_b64_str(Image.new("RGB", (100, 100), (255, 255, 255))),  # 创建一个白色背景图像
            "image2": get_image_b64_str(Image.new("RGB", (100, 100), (255, 0, 0)))  # 创建一个红色图像
        }
        response = requests.post(server_url, headers=headers, 
                                        data=json.dumps(demo_data), timeout=600)
        if "score" in response.json():
            return True
    except requests.RequestException:
        return False

# 初始化队列，添加可用服务端
def init_server_queue():
    servers = [
        "http://210.45.70.152:30930/compute_score",
        "http://210.45.70.152:30931/compute_score",
        # "http://127.0.0.1:30932/compute_score",
        # "http://127.0.0.1:30933/compute_score",
        # "http://127.0.0.1:30934/compute_score",
        # "http://127.0.0.1:30935/compute_score",
        # "http://127.0.0.1:30936/compute_score",
        # "http://127.0.0.1:30937/compute_score",
    ]
    print(servers)
    alive_servers = []
    for server in servers:
        if not test_server_connection(server):
            print(f"服务端 {server} 不可用，跳过添加到队列")
            continue
        print(f"服务端 {server} 可用，添加到队列")
        alive_servers.append(server)
    alive_servers = alive_servers * 8
    print(f"初始化服务端队列，共有 {len(alive_servers)} 个服务端")
    for server in alive_servers:
        SERVER_QUEUE.put(server)

@app.route('/compute_score', methods=['POST'])
def compute_score():
    # 获取客户端请求数据
    print(f"收到请求: ")
    data = request.get_json()
    text_content = data.get('prompt')
    svg_code = data.get('svg_code')
    bg_img_b64_str = data.get('image1')
    image_b64_str = data.get('image2')
    
    
    for _ in range(3):
        if True:
            # 尝试从队列中获取可用服务端
            server_url = SERVER_QUEUE.get()
            print(f"{datetime.now()} - 连接服务端: {server_url}")
            headers = {
                "Content-Type": "application/json",
            }
            
            request_data = {
                "prompt": text_content,
                "image1": bg_img_b64_str,
                "image2": image_b64_str,
            }
            
            response = requests.post(server_url, headers=headers, 
                                        data=json.dumps(request_data), timeout=600)
            
            request_data.update({"svg_code": svg_code})
            
            # 如果请求成功，将服务器重新加入队列尾部
            SERVER_QUEUE.put(server_url)
            
            result = response.json()
            ocr_result = get_ocr_metrics(request_data)
            result.update(ocr_result)
            
            
            print(f"Prompt: {text_content}", flush=True)
            print(result, flush=True)
            
            # 返回服务端响应
            if response.status_code == 200:
                return jsonify(result)
            else:
                print(f"服务端 {server_url} 返回非成功状态码: {response.status_code}")
                
        # except Exception as e:
        #     print(f"连接服务端 {server_url} 失败: {str(e)}")
        #     # 服务器失败，不重新加入队列
    
    return jsonify({"error": "所有服务端都不可用"}), 503

if __name__ == '__main__':
    # 初始化服务端队列
    init_server_queue()
    
    # 客户端统一访问的端口
    app.run(host='0.0.0.0', port=30949)