import re
import os
from datetime import datetime

class LRCEnhancer:
    def __init__(self):
        self.lyrics = []  # 每个元素: {'timestamp':, 'japanese':, 'chinese':, 'is_empty': bool}
    
    def parse_lrc(self, lrc_text):
        """解析LRC文件，保留所有时间戳行（包括空行）"""
        pattern = r'\[(\d{2}):(\d{2})\.(\d{2})\](.*?)(?=\[|$)'
        matches = re.findall(pattern, lrc_text, re.DOTALL)
        for match in matches:
            min_val, sec_val, cent_val, text = match
            timestamp = f"[{min_val}:{sec_val}.{cent_val}]"
            text = text.strip()
            # 保留该行，即使text为空
            self.lyrics.append({
                'timestamp': timestamp,
                'japanese': text,
                'chinese': '',
                'is_empty': (text == '')
            })
        return self
    
    def parse_bilingual_text(self, text):
        """按行顺序两两配对（日文行+中文行）"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        pairs = []
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                pairs.append({
                    'japanese': lines[i],
                    'chinese': lines[i + 1]
                })
            else:
                pairs.append({'japanese': lines[i], 'chinese': ''})
        return pairs
    
    def assign_translations(self, bilingual_pairs):
        """按顺序分配中文翻译到非空歌词行（跳过空行）"""
        # 先收集所有非空歌词行
        non_empty_indices = [i for i, item in enumerate(self.lyrics) if not item['is_empty']]
        # 只将翻译分配给这些行，按顺序
        for idx, pair in enumerate(bilingual_pairs):
            if idx < len(non_empty_indices):
                line_idx = non_empty_indices[idx]
                self.lyrics[line_idx]['chinese'] = pair['chinese']
        # 返回未分配翻译的歌词（如果有）
        unmatched = [item for item in self.lyrics if not item['is_empty'] and not item['chinese']]
        return unmatched
    
    def generate_enhanced_lrc(self):
        """生成增强LRC：非空行输出日文+中文，空行只输出时间戳"""
        output = []
        for item in self.lyrics:
            ts = item['timestamp']
            if item['is_empty']:
                # 空行只输出时间戳本身（没有文本）
                output.append(ts)
            else:
                output.append(f"{ts}{item['japanese']}")
                if item['chinese']:
                    output.append(f"{ts}{item['chinese']}")
        return '\n'.join(output)


def main():
    print("=" * 50)
    print("LRC增强生成器")
    print("=" * 50)
    
    file_path = input("\n请输入LRC文件路径: ").strip()
    file_path = file_path.strip('"').strip("'")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lrc_text = f.read()
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return
    
    enhancer = LRCEnhancer()
    enhancer.parse_lrc(lrc_text)
    
    # 统计非空歌词行数
    non_empty = [item for item in enhancer.lyrics if not item['is_empty']]
    empty_lines = [item for item in enhancer.lyrics if item['is_empty']]
    print(f"\n✅ 已读取LRC文件，共{len(enhancer.lyrics)}行（其中歌词{len(non_empty)}句，空行{len(empty_lines)}个）")
    
    # 显示LRC中的日文（方便对照）
    print("\nLRC中的日文歌词（非空行）：")
    print("-" * 50)
    for i, item in enumerate(non_empty, 1):
        print(f"{i:2d}. {item['japanese']}")
    if empty_lines:
        print(f"\n有空时间戳行：{', '.join([item['timestamp'] for item in empty_lines])}")
    print("-" * 50)
    
    print("\n请选择翻译输入方式：")
    print("1. 输入中日对照文本（日文行 + 中文行交替）")
    print("2. 只输入中文翻译（按顺序逐句输入）")
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == '1':
        print("\n请输入中日对照歌词（日文行 + 中文行交替）")
        print("输入完成后，在新行输入 'END' 结束\n")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            lines.append(line)
        text = '\n'.join(lines)
        pairs = enhancer.parse_bilingual_text(text)
        print(f"\n✅ 解析到 {len(pairs)} 组中日对照")
        
        unmatched = enhancer.assign_translations(pairs)
        matched = len(non_empty) - len(unmatched)
        print(f"✅ 已分配 {matched}/{len(non_empty)} 句翻译")
        
        if unmatched:
            print(f"\n⚠️ 有 {len(unmatched)} 句日文没有分配到中文翻译（输入行数不足）:")
            for item in unmatched:
                print(f"  - {item['japanese']}")
            confirm = input("\n是否继续生成? (y/n): ").strip().lower()
            if confirm != 'y':
                return
    else:
        print(f"\n请逐句输入中文翻译（共{len(non_empty)}句）")
        print("每行一句，输入完成后输入 'END' 结束\n")
        chinese_lines = []
        total = len(non_empty)
        while len(chinese_lines) < total:
            remaining = total - len(chinese_lines)
            print(f"还剩 {remaining} 句（第{len(chinese_lines)+1}-{total}句）:")
            line = input().strip()
            if line.upper() == 'END':
                break
            if line:
                chinese_lines.append(line)
        # 分配翻译给非空行
        non_empty_indices = [i for i, item in enumerate(enhancer.lyrics) if not item['is_empty']]
        for i, idx in enumerate(non_empty_indices):
            if i < len(chinese_lines):
                enhancer.lyrics[idx]['chinese'] = chinese_lines[i]
        matched = sum(1 for item in enhancer.lyrics if not item['is_empty'] and item['chinese'])
        if matched < total:
            print(f"\n⚠️ 警告: 只有 {matched}/{total} 句配对了中文翻译")
            confirm = input("是否继续生成? (y/n): ").strip().lower()
            if confirm != 'y':
                return
    
    # 预览最终结果
    print("\n歌词匹配预览（全部行）：")
    print("-" * 50)
    for item in enhancer.lyrics:
        if item['is_empty']:
            print(f"[空行] {item['timestamp']}")
        else:
            status = "✅" if item['chinese'] else "❌"
            print(f"{status} {item['japanese']}")
            if item['chinese']:
                print(f"   → {item['chinese']}")
    print("-" * 50)
    
    result = enhancer.generate_enhanced_lrc()

    # 输出到 output/ 文件夹，文件名沿用输入 LRC 的名字
    output_dir = os.path.join(os.path.dirname(file_path), 'output')
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = os.path.join(output_dir, f"{base_name}.lrc")
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(result)

    print(f"\n📁 输出文件夹: {output_dir}")
    print(f"📄 文件名: {base_name}.lrc")
    print(f"✅ 已生成: {output_filename}")
    print("\n输出预览（前20行）：")
    print("-" * 50)
    preview = result.split('\n')[:20]
    for line in preview:
        print(line)
    if len(result.split('\n')) > 20:
        print("...")
    print("-" * 50)


if __name__ == "__main__":
    main()