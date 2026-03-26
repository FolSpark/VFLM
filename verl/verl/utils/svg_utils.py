import os
import io
import base64
import megfile
import requests
from PIL import Image
from tqdm import tqdm

from multiprocessing import Pool
from lxml import etree
import xml.etree.ElementTree as ET
from playwright.sync_api import sync_playwright


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

def export_svg_to_img(svg_code: str, image: Image.Image, image_name: str = "background-image.png") -> Image.Image:
    try:
        svg_code = unformat_svg(svg_code)
    except Exception as e:
        print(f"[DEBUG] Error in unformat_svg: {e}")
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
    except Exception as e:
        print(f"[DEBUG] Error parsing SVG dimensions: {e}")
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
        # left = 8
        # top = 8
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
    svg_code = svg_code.replace("font-style=\"italic\"", "").replace("font-style=\"oblique\"", "")
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
            image = Image.open(io.BytesIO(image_bytes))
            return image
        else:
            print("调用失败", response.status_code, response.text)
            return None
    except Exception as e:
        print("处理异常:", e)
        return None


def worker(param):
    svg_path, output_dir, image_path, image_name = param
    if os.path.exists(os.path.join(output_dir, os.path.basename(svg_path).replace('.svg', '.png'))):
        # print(f"文件 {svg_path} 已存在，跳过处理。")
        return
    try:
        image = export_svg_to_img(svg_path, image_path=image_path, image_name=image_name)
        image.save(os.path.join(output_dir, os.path.basename(svg_path).replace('.svg', '.png')), format='PNG')
    except Exception as e:
        print(f"处理文件 {svg_path} 时出错: {str(e)}")
    



if __name__ == "__main__":
    # svg_path = "workspace/LayoutMLLM/svg_preprocess/formatted.svg"
    # output_dir = "workspace/LayoutMLLM/svg_preprocess/"
    # image_path = "workspace/LayoutMLLM/svg_preprocess/formatted-bg.png"
    # image_name = "background-image.png"
    
    # export_svg_to_png(svg_path, output_dir, image_path=image_path, image_name=None)
    
    # export_svg_to_png("data/datasets/svg-data/data/process_data/100006062-1.svg", "render_images")
    # export_svg_to_png("data/datasets/svg-data/data/process_data/100000075-1.svg", "render_images")
    # export_svg_to_png("unformatted.svg", "./")

    file_root = "workspace/datasets/svg-data/data/process_data_0424"
    svg_files = megfile.smart_glob(f"{file_root}/*.svg")
    # for file in tqdm(svg_files):
    #     # 处理每个SVG文件
    #     try:
    #         export_svg_to_png(file, file_root)
    #     except Exception as e:
    #         print(file)
    #         print(e)
    
    
    params = [(file, "workspace/datasets/svg-data/data/process_data_0424_img", "workspace/datasets/svg-data/data/process_data_0424/" + os.path.basename(file).replace(".svg", "-bg.png"), os.path.basename(file).replace(".svg", "-bg.png")) for file in svg_files]
    with Pool(64) as p:
        list(tqdm(p.imap(worker, params), total=len(svg_files)))