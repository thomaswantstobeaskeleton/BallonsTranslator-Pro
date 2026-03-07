import json
import os
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
from PIL import Image

# 全局变量，记录是否对所有图片使用同一尺寸
GLOBAL_PAGE_SIZE = None

def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception as e:
        return None

def prompt_for_size(image_name):
    global GLOBAL_PAGE_SIZE
    root = tk.Tk()
    root.withdraw()
    try:
        width = simpledialog.askinteger("输入图片宽度", f"无法读取图片 {image_name} 的尺寸，请输入宽度：", minvalue=1)
        height = simpledialog.askinteger("输入图片高度", f"无法读取图片 {image_name} 的尺寸，请输入高度：", minvalue=1)
        if width and height:
            # 询问是否应用于所有图片
            use_for_all = messagebox.askyesno("应用到所有图片", f"是否将此尺寸({width}x{height})应用于所有图片？")
            if use_for_all:
                GLOBAL_PAGE_SIZE = (width, height)
            return width, height
    except Exception:
        pass
    messagebox.showwarning("警告", f"输入无效，将使用默认尺寸 822*1200")
    return 822, 1200  # 修改为正确的默认尺寸

def get_image_info(json_data, image_name, json_file_path):
    """Get image dimensions from image_info, then from image files in project dir, else prompt."""
    image_info = json_data.get("image_info", {})
    if image_name in image_info:
        width = image_info[image_name].get("width")
        height = image_info[image_name].get("height")
        if width and height:
            return int(width), int(height)

    img_dir = os.path.dirname(json_file_path)
    img_base = os.path.splitext(image_name)[0]
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".jxl"]:
        candidate = os.path.join(img_dir, img_base + ext)
        if os.path.isfile(candidate):
            size = get_image_size(candidate)
            if size:
                return size

    size = prompt_for_size(image_name)
    return size

def extract_text_and_translation(json_file, include_font_info=False):
    global GLOBAL_PAGE_SIZE
    json_file_path = json_file

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_dir = os.path.dirname(json_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_text = []
    for image_name, items in data.get("pages", {}).items():
        PAGE_WIDTH, PAGE_HEIGHT = get_image_info(data, image_name, json_file_path)

        output_text.append(f">>>>>>>>[{image_name}]<<<<<<<<\n")

        for idx, item in enumerate(items):
            if "translation" not in item:
                continue
            coord = item.get("xyxy", [])
            if len(coord) >= 2:
                x, y = float(coord[0]), float(coord[1])
                coord = [x / PAGE_WIDTH, y / PAGE_HEIGHT, 1]
            else:
                coord = [0, 0, 1]

            translation = item["translation"] or ""

            if include_font_info:
                font_name = item.get("_detected_font_name") or ""
                font_size_val = item.get("_detected_font_size")
                if isinstance(font_size_val, (int, float)) and font_size_val > 0:
                    font_size_pt = font_size_val * 72 / 96
                    font_info = ""
                    if font_name:
                        font_info += f"{{字体：{font_name}}}"
                    font_info += f"{{字号：{font_size_pt}}}"
                    translation = font_info + translation
                elif font_name:
                    translation = f"{{字体：{font_name}}}" + translation

            output_text.append(f"----------------[{idx+1}]----------------{coord}\n")
            output_text.append(f"{translation}\n")

    json_base = os.path.splitext(os.path.basename(json_file))[0]
    output_file = os.path.join(output_dir, f"{json_base}_translations.txt")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write("1,0\n-\n框内\n框外\n-\n备注备注备注\n")
        f_out.writelines(output_text)

    print(f"翻译文本已保存到: {output_file}")
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("输出完成", f"翻译文本已成功输出到：\n{output_file}")

def choose_json_file():
    # 使用 tkinter 打开文件选择对话框
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    json_file = filedialog.askopenfilename(
        title="选择 JSON 文件",
        filetypes=[("JSON Files", "*.json")]
    )
    return json_file

def main():
    # 创建简单的GUI界面
    root = tk.Tk()
    root.title("气泡翻译器翻译转lptxt")
    root.geometry("400x200")
    
    # 添加复选框
    font_info_var = tk.BooleanVar()
    font_info_checkbox = tk.Checkbutton(root, text="导出时包含字体字号信息", variable=font_info_var)
    font_info_checkbox.pack(pady=20)
    
    # 添加处理按钮
    def process_json():
        # 用户选择 JSON 文件
        json_file = choose_json_file()
        if not json_file:
            messagebox.showwarning("警告", "未选择文件")
            return
        
        # 提取文本和翻译并保存到 txt 文件
        include_font_info = font_info_var.get()
        try:
            extract_text_and_translation(json_file, include_font_info)
        except Exception as e:
            messagebox.showerror("错误", f"处理文件时出错：{str(e)}")
        # 处理完成后不关闭窗口，继续等待下一次选择
    
    process_button = tk.Button(root, text="选择JSON文件并处理", command=process_json)
    process_button.pack(pady=10)
    
    # 添加退出按钮
    exit_button = tk.Button(root, text="退出", command=root.destroy)
    exit_button.pack(pady=10)
    
    # 运行GUI主循环
    root.mainloop()

if __name__ == "__main__":
    main()
