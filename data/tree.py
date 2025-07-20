import os

def print_tree(start_path, prefix=""):
    """
    递归打印目录树
    :param start_path: 要遍历的路径
    :param prefix: 当前缩进前缀
    """
    try:
        entries = sorted(os.listdir(start_path))
    except PermissionError:
        print(f"{prefix}└── [拒绝访问]")
        return

    # 过滤掉不需要显示的条目
    filtered_entries = []
    for entry in entries:
        full_path = os.path.join(start_path, entry)

        # 忽略 tree.py 自身
        if entry == 'tree.py':
            continue

        # 忽略以 . 开头的目录（隐藏目录）
        if entry.startswith('.') and os.path.isdir(full_path):
            continue

        # 忽略形如 __xxx__ 的目录，但不是文件
        if entry.startswith('__') and entry.endswith('__') and os.path.isdir(full_path):
            continue

        # 忽略 .png 和 .jpg 文件
        if entry.lower().endswith(('.png', '.jpg')):
            continue

        filtered_entries.append(entry)

    entries = filtered_entries

    for i, entry in enumerate(entries):
        path = os.path.join(start_path, entry)
        is_last = i == len(entries) - 1
        new_prefix = "    " if is_last else "│   "

        connector = "└──" if is_last else "├──"
        print(f"{prefix}{connector} {entry}")

        if os.path.isdir(path):
            print_tree(path, prefix + new_prefix)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_name = os.path.basename(current_dir.rstrip(os.sep)) + '/'  # 获取当前目录名并加斜杠
    print(f"{dir_name}")
    print_tree(current_dir)