import re
import os
import io
import cv2
import math
import base64
import megfile
import requests
import numpy as np
from tqdm import tqdm
from typing import Tuple
from PIL import Image, ImageDraw

from lxml import etree
import xml.etree.ElementTree as ET
from playwright.sync_api import sync_playwright
from multiprocessing import Pool


def extract_svg_text(svg_code, return_list=False):
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
    if return_list:
        return [' '.join(part.split()) for part in all_text]
    else:
        return ' '.join(' '.join(part.split()) for part in all_text)


def extract_font_families(svg_content):
    """从 SVG 内容中提取所有 font-family 值"""
    # 用于匹配 font-family 属性的正则表达式
    font_family_regex = r'font-family\s*=\s*["\']([^"\']+)["\']'
    matches = re.findall(font_family_regex, svg_content)
    
    # 处理可能包含多个字体的情况
    font_families = []
    for match in matches:
        # 分割多个字体（使用逗号分隔）
        fonts = [font.strip() for font in match.split(',')]
        font_families.extend(fonts)
    
    # 去重并返回
    return list(set(font_families))


def remove_namespace(element):
    """安全移除元素及其子元素的命名空间前缀"""
    for elem in element.iter():
        # 修复元素标签（确保是字符串类型）
        if isinstance(elem.tag, str) and '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
        # 修复属性命名空间
        for attr in list(elem.attrib):
            if isinstance(attr, str) and '}' in attr:
                new_attr = attr.split('}', 1)[1]
                elem.attrib[new_attr] = elem.attrib[attr]
                del elem.attrib[attr]
    return element


def unformat_svg(svg_code):
    # 定义命名空间映射
    namespaces = {
        'svg': 'http://www.w3.org/2000/svg',
        'xlink': 'http://www.w3.org/1999/xlink'
    }
    
    # 使用正确解析器
    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    # tree = etree.parse(svg_path, parser)
    # root = tree.getroot()
    root = etree.fromstring(svg_code, parser)
    
    # 修复根元素命名空间
    root.tag = etree.QName(root).localname
    
    # 清理命名空间
    root = remove_namespace(root)
    
    # 修复可能的属性类型错误
    for elem in root.iter():
        # 清理属性值中的非字符串类型
        for attr in list(elem.attrib):
            if not isinstance(elem.attrib[attr], str):
                elem.attrib[attr] = str(elem.attrib[attr])
    
    # 创建新树并保存（禁用格式化）
    new_tree = etree.ElementTree(root)
    xml_content = etree.tostring(
        new_tree,
        pretty_print=False,
        xml_declaration=False,
        encoding='utf-8',
        method='xml',
        with_tail=False,
    ).decode('utf-8')
    
    # 最终清理（移除可能残留的xmlns声明）
    xml_content = xml_content.replace('xmlns="http://www.w3.org/2000/svg"', '')
    # print(xml_content)
    
    return xml_content


def export_svg_to_img(svg_code: str, image: Image.Image, image_name: str = "background-image.png", unformat: bool = True) -> Image.Image:
    """
    使用Playwright将SVG代码渲染为图片，返回PIL.Image对象。非服务
    """
    # svg_code = open(svg_path, 'r').read()
    # svg_code = unformat_svg(svg_path)
    if unformat:
        try:
            svg_code = unformat_svg(svg_code)
        except Exception as e:
            print(f"Error unformatting SVG: {e}")
            print(f"Using original SVG code instead.")
    # 图片相对路径改为base64
    # image_path = svg_path.replace('.svg', '-bg.png') if image_path is None else image_path
    # image_name = os.path.basename(image_path) if image_name is None else image_name
    # image = Image.open(svg_path.replace('.svg', '-bg.png'))
    # image = Image.open(image_path)
    image_data = io.BytesIO()
    image.save(image_data, format='PNG')
    image_base64 = base64.b64encode(image_data.getvalue()).decode('utf-8')
    svg_code = svg_code.replace(image_name, f'data:image/png;base64,{image_base64}')
    try:
        root = ET.fromstring(svg_code)
        body_width = int(float(root.get("width")))
        body_height = int(float(root.get("height")))
        view_box = root.get("viewBox")
    except:
        body_width = image.width
        body_height = image.height
    
    with sync_playwright() as p:
        # 启动无头Chromium浏览器
        browser = p.chromium.launch(headless=True, args=["--allow-file-access-from-files"])
        page = browser.new_page()
        
        # print(f"body_width: {body_width}, body_height: {body_height}")
        page.set_viewport_size({"width": body_width, "height": body_height})
        # 加载HTML内容
        page.set_content(svg_code, timeout=60000)
        # 等待页面加载完成
        page.wait_for_load_state("networkidle")
        # 截取完整页面并保存
        # page.screenshot(path=output_path, full_page=True)
        
        # Capture screenshot as bytes
        # screenshot_bytes = page.screenshot(full_page=True)
        try:
            svg_handle = page.wait_for_selector("svg", timeout=5000)
            screenshot_bytes = svg_handle.screenshot(timeout=10000)
        except Exception:
            print("SVG element not found, falling back to full page screenshot.")
            # fallback：整页 clip
            screenshot_bytes = page.screenshot(
                clip={'x': 8, 'y': 8, 'width': body_width + 8, 'height': body_height + 8}
            )
        # Convert to PIL Image for post-processing
        image = Image.open(io.BytesIO(screenshot_bytes))
        browser.close()
        
        # Here you can do post-processing on the image
        # For example:
        # 截图区域
        # left = 0
        # top = 0
        # right = body_width + 8
        # bottom = body_height + 8
        # # Crop the image to the specified area
        # image = image.crop((left, top, right, bottom))
        
        return image


def export_pure_svg_to_img(svg_code: str) -> Image.Image:
    root = ET.fromstring(svg_code)
    body_width = int(float(root.get("width")))
    body_height = int(float(root.get("height")))
    view_box = root.get("viewBox")
    
    with sync_playwright() as p:
        # 启动无头Chromium浏览器
        browser = p.chromium.launch(headless=True, args=["--allow-file-access-from-files"])
        page = browser.new_page()
        
        # print(f"body_width: {body_width}, body_height: {body_height}")
        page.set_viewport_size({"width": body_width, "height": body_height})
        # 加载HTML内容
        page.set_content(svg_code, timeout=60000)
        # 等待页面加载完成
        page.wait_for_load_state("networkidle")
        # 截取完整页面并保存
        # page.screenshot(path=output_path, full_page=True)
        
        # Capture screenshot as bytes
        # screenshot_bytes = page.screenshot(full_page=True)
        try:
            svg_handle = page.wait_for_selector("svg", timeout=5000)
            screenshot_bytes = svg_handle.screenshot(timeout=10000)
        except Exception:
            print("SVG element not found, falling back to full page screenshot.")
            # fallback：整页 clip
            screenshot_bytes = page.screenshot(
                clip={'x': 8, 'y': 8, 'width': body_width + 8, 'height': body_height + 8}
            )
        # Convert to PIL Image for post-processing
        image = Image.open(io.BytesIO(screenshot_bytes))
        browser.close()
        
        # Here you can do post-processing on the image
        # For example:
        # 截图区域
        # left = 0
        # top = 0
        # right = body_width + 8
        # bottom = body_height + 8
        # # Crop the image to the specified area
        # image = image.crop((left, top, right, bottom))
        
        return image


def export_svg_with_bg(svg_code: str, background_image: Image.Image, api_url: str = "http://10.0.2.226:8999/export_svg", unformat: bool = True) -> 'Image.Image|None':
    """
    调用服务将SVG和背景图片合成，返回PIL.Image对象。
    :param svg_code: SVG代码字符串
    :param background_image: 已经用PIL.Image读取的背景图片
    :param api_url: 服务API地址
    :return: PIL.Image对象或None
    """
    if unformat:
        try:
            svg_code = unformat_svg(svg_code)
        except Exception as e:
            print(f"Unformat SVG failed: {e}")

    # if True:
    try:
        buf = io.BytesIO()
        background_image.save(buf, format="PNG")
        background_image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "svg_code": svg_code,
            "bg_image_base64": background_image_base64
        }

        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                print("错误信息:", result["error"])
                return None
            output_base64 = result["image_base64"]
            image_bytes = base64.b64decode(output_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return image
        else:
            print("调用失败", response.status_code, response.text)
            return None
    except Exception as e:
        print("处理异常:", e)
        return None


# ---------- 工具函数 ----------
def _parse_number(s: str) -> float:
    """解析SVG中的数字（忽略单位）"""
    s = s.strip()
    if s.endswith(('px', 'pt', 'em', 'rem')):
        return float(s[:-2])
    return float(s)


def _svg_matrix(elem) -> np.ndarray:
    """计算元素的累积变换矩阵（3x3齐次矩阵）"""
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    trans = elem.get('transform', '')
    parent = elem.getparent()
    
    # 递归获取父元素的变换矩阵
    m = np.eye(3)
    if parent is not None and parent.tag.startswith('{' + ns['svg']):
        m = _svg_matrix(parent)

    # 解析当前元素的变换
    for t in re.finditer(r'(\w+)\(([^)]+)\)', trans):
        op, raw = t.group(1), t.group(2)
        args = list(map(float, re.split(r'[\s,]+', raw.strip())))
        
        if op == 'translate':
            dx, dy = (args + [0])[:2]
            m = m @ np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
        elif op == 'scale':
            sx, sy = (args + [args[0]])[:2]
            m = m @ np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        elif op == 'rotate':
            a = math.radians(args[0])
            if len(args) == 3:
                cx, cy = args[1], args[2]
                m = m @ np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
                m = m @ np.array([[math.cos(a), -math.sin(a), 0],
                                  [math.sin(a),  math.cos(a), 0],
                                  [0, 0, 1]])
                m = m @ np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
            else:
                m = m @ np.array([[math.cos(a), -math.sin(a), 0],
                                  [math.sin(a),  math.cos(a), 0],
                                  [0, 0, 1]])
    return m


def _get_text_transform_info(elem) -> Tuple[float, float, float, np.ndarray]:
    """
    提取文本元素的变换信息
    返回: (旋转角度(弧度), 旋转中心点x, 旋转中心点y, 不含旋转的变换矩阵)
    """
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    trans = elem.get('transform', '')
    
    rotation = 0.0
    cx, cy = 0.0, 0.0
    rotation_matrix = np.eye(3)
    
    for t in re.finditer(r'(\w+)\(([^)]+)\)', trans):
        op, raw = t.group(1), t.group(2)
        args = list(map(float, re.split(r'[\s,]+', raw.strip())))
        
        if op == 'rotate':
            rotation = math.radians(args[0])
            if len(args) == 3:
                cx, cy = args[1], args[2]
                # 构建旋转矩阵（含中心点）
                rotation_matrix = (
                    np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]]) @
                    np.array([[math.cos(rotation), -math.sin(rotation), 0],
                              [math.sin(rotation),  math.cos(rotation), 0],
                              [0, 0, 1]]) @
                    np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
                )
            else:
                # 构建旋转矩阵（无中心点）
                rotation_matrix = np.array([
                    [math.cos(rotation), -math.sin(rotation), 0],
                    [math.sin(rotation),  math.cos(rotation), 0],
                    [0, 0, 1]
                ])
    
    # 构建不含旋转的变换矩阵
    non_rotation_matrix = np.eye(3)
    for t in re.finditer(r'(\w+)\(([^)]+)\)', trans):
        op, raw = t.group(1), t.group(2)
        args = list(map(float, re.split(r'[\s,]+', raw.strip())))
        
        if op != 'rotate':  # 跳过旋转操作
            if op == 'translate':
                dx, dy = (args + [0])[:2]
                non_rotation_matrix = non_rotation_matrix @ np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
            elif op == 'scale':
                sx, sy = (args + [args[0]])[:2]
                non_rotation_matrix = non_rotation_matrix @ np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
    
    # 合并父级变换
    parent = elem.getparent()
    if parent is not None and parent.tag.startswith('{' + ns['svg']):
        parent_rot, parent_cx, parent_cy, parent_non_rot_matrix = _get_text_transform_info(parent)
        
        # 合并旋转角度
        rotation += parent_rot
        
        # 合并不含旋转的变换矩阵
        non_rotation_matrix = parent_non_rot_matrix @ non_rotation_matrix
        
        # 父级变换会影响子级旋转中心点
        if parent_rot != 0 and (cx != 0 or cy != 0):
            # 应用父级旋转变换
            cos_p = math.cos(parent_rot)
            sin_p = math.sin(parent_rot)
            dx = cx - parent_cx
            dy = cy - parent_cy
            cx = parent_cx + dx * cos_p - dy * sin_p
            cy = parent_cy + dx * sin_p + dy * cos_p
    
    return rotation, cx, cy, non_rotation_matrix


def _modify_text_color(text_elem):
    """强制文本颜色为黑色（确保与白色背景对比）"""
    text_elem.set('fill', 'black')
    if 'style' in text_elem.attrib:
        style = text_elem.get('style')
        style = re.sub(r'fill\s*:\s*[^;]+;?', '', style)  # 移除原有fill样式
        text_elem.set('style', style.strip())
    return text_elem


def _extract_non_white_bbox(img: Image.Image) -> Tuple[int, int, int, int]:
    """从图片中提取非白色区域的边界框（x,y,w,h）"""
    arr = np.array(img)
    # 判断非白色像素（允许轻微偏差）
    non_white = ~np.all(np.isclose(arr, [255, 255, 255], atol=10), axis=2)
    if not np.any(non_white):
        return 0, 0, 0, 0
    # 获取非白色区域的最小/最大坐标
    rows, cols = np.where(non_white)
    ymin, ymax = rows.min(), rows.max()
    xmin, xmax = cols.min(), cols.max()
    return int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)


def _transform_points(x: int, y: int, w: int, h: int, matrix: np.ndarray) -> np.ndarray:
    """将局部边界框的4个角点通过变换矩阵映射到全局坐标"""
    # 局部边界框的4个角点
    corners = np.array([
        [x, y, 1],
        [x + w, y, 1],
        [x + w, y + h, 1],
        [x, y + h, 1]
    ])
    # 应用变换矩阵得到全局坐标
    return (matrix @ corners.T).T[:, :2]


def all_text_bboxes(labels_svg: str) -> list[np.ndarray]:
    """在原图上绘制所有文本的边界框（对90度旋转文本特殊处理）"""
    # 解析SVG
    tree = etree.ElementTree(etree.fromstring(labels_svg))
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    svg_width = _parse_number(root.get('width', '400'))
    svg_height = _parse_number(root.get('height', '200'))
    
    # 获取SVG的viewBox属性（如果有）
    viewBox = root.get('viewBox', '')
    if viewBox:
        viewBox = list(map(float, viewBox.split()))
        vb_x, vb_y, vb_width, vb_height = viewBox
        # 计算viewBox到实际尺寸的变换矩阵
        scale_x = svg_width / vb_width
        scale_y = svg_height / vb_height
        viewBox_matrix = np.array([
            [scale_x, 0, -vb_x * scale_x],
            [0, scale_y, -vb_y * scale_y],
            [0, 0, 1]
        ])
    else:
        viewBox_matrix = np.eye(3)

    global_corners_list = []  # 用于存储所有文本的全局边界框坐标
    # 处理每个文本元素
    for text in root.xpath('//svg:text', namespaces=ns):
        # 获取文本的变换信息
        rotation, cx, cy, non_rotation_matrix = _get_text_transform_info(text)
        angle_deg = abs(math.degrees(rotation) % 360)
        # is_90_deg_rotation = abs(angle_deg - 90) < 2 or abs(angle_deg - 270) < 2
        is_90_deg_rotation = abs(angle_deg % 90) < 2
        
        # 1. 复制文本元素（避免修改原始DOM）
        text_copy = etree.fromstring(etree.tostring(text))
        text_copy = _modify_text_color(text_copy)
        
        # 2. 获取文本的定位点
        text_x = _parse_number(text.get('x', '0'))
        text_y = _parse_number(text.get('y', '0'))
        
        if is_90_deg_rotation:
            # 90度旋转文本：保留原始处理（渲染已旋转的文本）
            # 创建仅包含当前文本的SVG（带旋转）
            single_svg = etree.Element('svg', nsmap={None: 'http://www.w3.org/2000/svg'})
            single_svg.set('width', str(svg_width))
            single_svg.set('height', str(svg_height))
            single_svg.set('viewBox', f'0 0 {svg_width} {svg_height}')
            single_svg.append(text_copy)

            # 渲染SVG并提取边界框
            svg_code = etree.tostring(single_svg).decode('utf-8')
            white_img = Image.new('RGBA', (int(svg_width), int(svg_height)), (255, 255, 255, 255))
            img = export_svg_to_img(svg_code, white_img)
            # img = export_svg_with_bg(svg_code, white_img)
            x, y, w, h = _extract_non_white_bbox(img)
            if w == 0 or h == 0:
                continue
            
            # 应用不含旋转的变换矩阵（平移和缩放）
            corners = np.array([
                [x, y, 1],
                [x + w, y, 1],
                [x + w, y + h, 1],
                [x, y + h, 1]
            ])
            
            # 应用不含旋转的变换矩阵和viewBox变换
            full_non_rot_matrix = viewBox_matrix @ non_rotation_matrix
            global_corners = (full_non_rot_matrix @ corners.T).T[:, :2]
            global_corners_list.append(global_corners)
            
        else:
            # 对于非90度旋转的文本：使用原始方法（应用完整变换矩阵）
            # 1. 创建仅包含当前文本的SVG
            single_svg = etree.Element('svg', nsmap={None: 'http://www.w3.org/2000/svg'})
            single_svg.set('width', str(svg_width))
            single_svg.set('height', str(svg_height))
            single_svg.set('viewBox', f'0 0 {svg_width} {svg_height}')
            # 复制文本元素并强制黑色
            # 非90度旋转文本：先提取未旋转的紧凑边界框，再应用变换
            # 关键修改：移除文本元素的变换属性，渲染未旋转的原始文本
            original_transform = text_copy.get('transform', '')
            if original_transform:
                del text_copy.attrib['transform']  # 临时移除变换，获取原始文本形态
            single_svg.append(text_copy)

            # 2. 用浏览器渲染SVG
            svg_code = etree.tostring(single_svg).decode('utf-8')
            white_img = Image.new('RGBA', (int(svg_width), int(svg_height)), (255, 255, 255, 255))
            img = export_svg_to_img(svg_code, white_img)

            # 3. 提取渲染后非白色区域的边界框
            x, y, w, h = _extract_non_white_bbox(img)
            if w == 0 or h == 0:
                continue  # 跳过空文本

            # 4. 计算文本元素的全局变换矩阵
            matrix = _svg_matrix(text)

            # 5. 将局部边界框映射到全局坐标
            global_corners = _transform_points(x, y, w, h, matrix)
            global_corners_list.append(global_corners)

    return global_corners_list


def extract_skewed_rectangle(image: Image.Image, global_corners: list[np.ndarray]) -> Image.Image:
    """
    从图像中提取倾斜的矩形区域
    
    参数:
    image (PIL.Image): 输入图像
    global_corners (np.ndarray): 矩形的四个角点坐标，形状为(4, 2)
    
    返回:
    PIL.Image: 提取出的矩形区域
    """
    # 获取四个角点坐标
    pts_src = np.float32(global_corners)
    
    # 计算目标矩形的宽度和高度
    width_a = np.sqrt(((pts_src[0][0] - pts_src[1][0]) ** 2) + ((pts_src[0][1] - pts_src[1][1]) ** 2))
    width_b = np.sqrt(((pts_src[2][0] - pts_src[3][0]) ** 2) + ((pts_src[2][1] - pts_src[3][1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.sqrt(((pts_src[0][0] - pts_src[3][0]) ** 2) + ((pts_src[0][1] - pts_src[3][1]) ** 2))
    height_b = np.sqrt(((pts_src[1][0] - pts_src[2][0]) ** 2) + ((pts_src[1][1] - pts_src[2][1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    # 定义目标矩形的四个角点坐标
    pts_dst = np.float32([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ])
    
    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    
    # 应用透视变换
    result = cv2.warpPerspective(np.array(image), matrix, (max_width, max_height))
    
    # 转换回PIL Image
    return Image.fromarray(result)
