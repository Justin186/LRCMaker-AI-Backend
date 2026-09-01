import os
import sys
import json
import socket
import asyncio
import itertools
import tempfile
import datetime
import warnings
import re
import multiprocessing
from contextlib import asynccontextmanager
from types import SimpleNamespace

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import stable_whisper
from stable_whisper.audio import load_audio

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 禁用 huggingface 的 xet 加速下载（在部分网络/Windows 环境下会报 401 错误）
os.environ["HF_HUB_DISABLE_XET"] = "1"

if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
    env_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else application_path
    os.environ["PATH"] = env_path + os.pathsep + os.environ.get("PATH", "")
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

model = None

# ========== 可配置项 ==========
# 模型规格：'tiny' / 'base' / 'small' / 'medium' / 'large-v3'
# 越大越准但越慢/越占显存。GPU 上推荐 medium 或 large-v3
MODEL_SIZE = "small"
# 是否强制使用 GPU。True=优先 GPU(无则回退CPU)，False=强制 CPU
FORCE_GPU = True
# 词间停顿判定阈值（秒）：相邻两词间隔超过该值，视为句中停顿
GAP_THRESHOLD = 0.001
# 是否记录请求日志（源文本 + 输出 LRC 写入 logs/align_requests.log）。True=记录，False=不记录
ENABLE_REQUEST_LOG = True
# 是否把上传的音频保存到项目 audio/ 目录（方便试听/调试）。True=保存，False=不保存
SAVE_AUDIO = True
# =============================

# ===== 自定义警告处理器：美化 stable_whisper 的对齐警告输出 =====
_original_showwarning = warnings.showwarning
_warning_buffer = []

def _beautified_showwarning(message, category, filename, lineno, file=None, line=None):
    """将 stable_whisper 的原始 Python 警告重新格式化为简洁美观的输出"""
    msg = str(message)
    # 只拦截来自 stable_whisper.alignment 的警告
    if "stable_whisper" in (filename or ""):
        # 匹配 "Failed to align the last N/M words after HH:MM:SS."
        m = re.match(r"Failed to align the last (\d+)/(\d+) words after (\d+:\d+:\d+\.\d+)\.", msg)
        if m:
            failed, total, ts = m.group(1), m.group(2), m.group(3)
            print(f"  ⚠️  对齐警告: 尾部 {failed}/{total} 个词在 {ts} 后未能对齐（已跳过）")
            return
        # 匹配 "N/M segments failed to align."
        m = re.match(r"(\d+)/(\d+) segments? failed to align\.", msg)
        if m:
            failed, total = m.group(1), m.group(2)
            print(f"  ⚠️  对齐警告: {failed}/{total} 个分段未能对齐（已跳过）")
            return
    # 其他警告保持原样
    _original_showwarning(message, category, filename, lineno, file, line)

warnings.showwarning = _beautified_showwarning

# 请求计数器：用于检测并丢弃过时请求的结果
_next_request_id = itertools.count(1)
_current_request_id = None

def pick_device():
    """自动选择计算设备：优先 CUDA GPU，否则回退 CPU"""
    try:
        import torch
        if FORCE_GPU and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🎮 检测到 NVIDIA GPU: {gpu_name}，使用 GPU 加速推理！")
            return "cuda"
    except Exception as e:
        print(f"⚠️ 无法加载 torch 检测 GPU: {e}")
    print("💻 未检测到可用 GPU，回退到 CPU 模式（速度较慢）。")
    return "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("🚀 正在初始化 AI 引擎...")
    
    model_cache_dir = os.path.expanduser("~/.lrc_maker_models")
    os.makedirs(model_cache_dir, exist_ok=True)
    
    # 优先使用本地离线模型目录（release 打包时内置）
    local_model_path = os.path.join(application_path, "models", f"faster-whisper-{MODEL_SIZE}")
    
    if os.path.exists(local_model_path) and os.listdir(local_model_path):
        print(f"📦 发现本地离线模型目录，直接加载免下载：\n   -> {local_model_path}")
        model_target = local_model_path
        is_local_model = True
    else:
        print(f"🌐 未发现本地离线模型，将尝试从网络或系统缓存加载 '{MODEL_SIZE}' 模型...")
        print(f"   (提示：你可以手动下载模型并放入 {local_model_path} 目录实现完全离线运行)")
        model_target = MODEL_SIZE
        is_local_model = False
    
    device = pick_device()
    
    model = stable_whisper.load_faster_whisper(
        model_target, 
        device=device, 
        compute_type="float16" if device == "cuda" else "int8",
        download_root=model_cache_dir
    )

    # 首次从网络下载后，自动复制到项目 models/ 目录，方便下次离线加载
    if not is_local_model:
        import shutil
        # huggingface_hub 缓存结构：~/.lrc_maker_models/models--<org>--<name>/snapshots/<hash>/
        # 需要从 snapshots 子目录里找到实际模型文件
        cache_root = os.path.join(model_cache_dir, f"models--Systran--faster-whisper-{MODEL_SIZE}")
        downloaded_path = None
        snapshots_dir = os.path.join(cache_root, "snapshots")
        if os.path.isdir(snapshots_dir):
            for entry in os.listdir(snapshots_dir):
                candidate = os.path.join(snapshots_dir, entry)
                if os.path.isdir(candidate) and os.listdir(candidate):
                    downloaded_path = candidate
                    break
        if downloaded_path and os.path.exists(downloaded_path):
            try:
                print(f"📦 正在将模型复制到本地目录以供离线使用：\n   -> {local_model_path}")
                # symlinks=False（默认）：跟随符号链接复制实际文件内容，
                # 避免复制出指向 blobs 的无效链接
                shutil.copytree(downloaded_path, local_model_path, dirs_exist_ok=True, symlinks=False)
                print(f"✅ 模型已保存到本地，下次启动将直接加载，无需联网。")
            except Exception as e:
                print(f"⚠️ 复制模型到本地目录失败（不影响本次运行）: {e}")
        else:
            print(f"⚠️ 未在缓存中找到模型文件（{cache_root}），跳过复制。")

    print("✅ AI 引擎就绪！请不要关闭此黑色窗口。")
    print("👉 现在去网页里点击 [一键 AI 强制对齐] 吧！")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

def format_time(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"[{minutes:02d}:{remaining_seconds:05.2f}]"

def log_request(audio_name: str, source_text: str, result: dict):
    """将每次对齐请求的源文本与输出 LRC 写入日志文件"""
    if not ENABLE_REQUEST_LOG:
        return
    try:
        log_dir = os.path.join(application_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "align_requests.log")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"[{timestamp}] 音频文件: {audio_name}\n")
            f.write("-" * 60 + "\n")
            f.write("[源文本]\n")
            f.write(source_text + "\n")
            f.write("-" * 60 + "\n")
            # # 原始传入数据调试信息（每行歌词 + 参考时间戳），便于排查对齐问题
            # debug_info = result.get("debug_info", "")
            # if debug_info:
            #     f.write("[原始传入数据]\n")
            #     f.write(debug_info + "\n")
            #     f.write("-" * 60 + "\n")
            f.write("[标准 LRC]\n")
            f.write(result.get("standard_lrc", "") + "\n")
            f.write("-" * 60 + "\n")
            f.write("[逐字 LRC]\n")
            f.write(result.get("enhanced_lrc", "") + "\n")
            f.write("=" * 60 + "\n\n")
        print(f"📝 请求日志已写入: {log_path}")
    except Exception as e:
        print(f"⚠️ 写入请求日志失败: {e}")

def clean_str(s):
    """去除所有不可见空白字符，用于歌词行与模型输出词的匹配。

    注意：除了普通空格、全角空格、换行、回车、制表符外，还必须去除
    U+00A0（不换行空格 NBSP）、U+200B（零宽空格）等 Unicode 空白字符。
    实测发现网易云歌词中可能混入 NBSP（如「どうやら私ら\xa0久しく...」），
    若不去除会导致歌词行 target 与模型输出词无法匹配，进而整段对齐失败。
    """
    return (s.replace(" ", "")
             .replace("\u3000", "")   # 全角空格
             .replace("\u00a0", "")   # 不换行空格 NBSP
             .replace("\u200b", "")   # 零宽空格
             .replace("\u2009", "")   # 窄空格
             .replace("\u2002", "")   # 半角空格
             .replace("\u2003", "")   # 全角空格(em)
             .replace("\n", "")
             .replace("\r", "")
             .replace("\t", ""))

def split_line_to_segments(line):
    """将歌词行拆分为匹配段，用于逐字对齐。

    CJK 汉字（中日）和日文假名（平假名/片假名）每个字符单独作为一个段，
    韩文（Hangul 音节）和拉丁文等按空格分词。

    返回 (segments, end_positions)：
    - segments: 拆分后的文本段列表
    - end_positions: 每段在原始行中的结束位置（用于判断段间是否有空格）

    示例:
        "天下相亲与相爱" → ["天","下","相","亲","与","相","爱"], [1,2,3,4,5,6,7]
        "If I send"      → ["If","I","send"], [2,4,8]
        "天下相爱 动身"   → ["天","下","相","爱","动","身"], [1,2,3,4,6,7]
                                                    ↑ 第4段"爱"结束于位置4，line[4]=' ' → 有空格
    """
    segments = []
    end_positions = []
    current = ""
    for i, char in enumerate(line):
        if char == ' ' or char == '\u3000':
            # 空格/全角空格：保存当前段，记录结束位置在空格之前
            if current:
                segments.append(current)
                end_positions.append(i)
                current = ""
        elif ('\u4e00' <= char <= '\u9fff' or  # CJK 统一汉字（中日共用）
              '\u3400' <= char <= '\u4dbf' or  # CJK 扩展 A
              '\u3040' <= char <= '\u309f' or  # 平假名
              '\u30a0' <= char <= '\u30ff'):    # 片假名
            # CJK 字符：保存当前段，然后单独作为一个段
            if current:
                segments.append(current)
                end_positions.append(i)
                current = ""
            segments.append(char)
            end_positions.append(i + 1)
        else:
            # 韩文、拉丁文、数字、标点等：累积到当前段
            current += char
    if current:
        segments.append(current)
        end_positions.append(len(line))
    return segments, end_positions

def detect_audio_ext(content: bytes) -> str:
    """根据文件头（magic bytes）识别音频格式，返回带点的扩展名（如 '.mp3'）。
    无法识别时返回 None。"""
    if not content:
        return None
    # MP3: ID3 标签开头，或 0xFF 0xFB/0xF3/0xF2 等帧同步
    if content[:3] == b'ID3' or (content[0] == 0xFF and (content[1] & 0xE0) == 0xE0):
        return '.mp3'
    # WAV/RIFF
    if content[:4] == b'RIFF' and content[8:12] == b'WAVE':
        return '.wav'
    # FLAC
    if content[:4] == b'fLaC':
        return '.flac'
    # OGG
    if content[:4] == b'OggS':
        return '.ogg'
    # M4A/MP4 (ftyp)
    if content[4:8] == b'ftyp':
        return '.m4a'
    # AAC (ADTS)
    if content[0] == 0xFF and (content[1] & 0xF6) == 0xF0:
        return '.aac'
    # WMA (ASF)
    if content[:16] == b'\x30\x26\xb2\x75\x8e\x66\xcf\x11\xa6\xd9\x00\xaa\x00\x62\xce\x6c':
        return '.wma'
    return None

def detect_language(text: str) -> str:
    """根据歌词内容自动判断语言，返回 faster-whisper 支持的语言代码"""
    import re
    # 统计各类字符数量
    han_count = len(re.findall(r'[\u4e00-\u9fff]', text))          # 简体中文
    hant_count = len(re.findall(r'[\u3400-\u4dbf]', text))          # 扩展区汉字
    kana_count = len(re.findall(r'[\u3040-\u30ff]', text))          # 日文假名
    hangul_count = len(re.findall(r'[\uac00-\ud7af]', text))        # 韩文
    latin_count = len(re.findall(r'[A-Za-z]', text))                # 拉丁字母（英/法/德等）
    cyrillic_count = len(re.findall(r'[\u0400-\u04ff]', text))      # 西里尔字母（俄语等）
    thai_count = len(re.findall(r'[\u0e00-\u0e7f]', text))          # 泰文

    # 按字符数占比判断
    total = max(1, han_count + hant_count + kana_count + hangul_count + latin_count + cyrillic_count + thai_count)

    if kana_count / total > 0.3:
        return 'ja'
    if hangul_count / total > 0.3:
        return 'ko'
    if (han_count + hant_count) / total > 0.3:
        return 'zh'
    if cyrillic_count / total > 0.3:
        return 'ru'
    if thai_count / total > 0.3:
        return 'th'
    if latin_count / total > 0.3:
        return 'en'
    # 默认回退中文
    return 'zh'

def generate_lrc_content(audio_path: str, raw_lyrics_text: str, ti: str, ar: str, al: str, times=None) -> dict:
    raw_lines = raw_lyrics_text.splitlines()
    staff_lines = []
    sung_lines = []
    sung_indices = []  # sung_lines 每行在 raw_lines 中的索引
    has_separator = any(line.strip() == "---" for line in raw_lines)
    is_sung_started = False
    
    for idx, line in enumerate(raw_lines):
        line = line.strip()
        if not line: continue
        if has_separator:
            if line == "---":
                is_sung_started = True
                continue
            if not is_sung_started:
                staff_lines.append(line)
            else:
                sung_lines.append(line)
                sung_indices.append(idx)
        else:
            if not is_sung_started and (":" in line or "：" in line):
                staff_lines.append(line)
            else:
                is_sung_started = True
                sung_lines.append(line)
                sung_indices.append(idx)
                
    if not sung_lines:
        return {"error": "❌ 错误：未在文本中找到有效歌词，请检查排版。"}

    full_text = "\n".join(sung_lines)

    # 纯音乐检测：歌词文本为"纯音乐，请欣赏"等占位符时，无需对齐
    pure_music_patterns = ["纯音乐", "纯音樂", " instrumental", "（伴奏）", "(伴奏)", "暂无歌词"]
    if any(pat in full_text for pat in pure_music_patterns):
        return {"error": "❌ 这是一首纯音乐，没有歌词可供对齐。"}

    lang = detect_language(full_text)
    print(f"🌐 检测到歌词语言: {lang}")

    # 解析参考时间戳（秒），对应 sung_lines 每行
    ref_times = None
    if times and len(times) == len(raw_lines):
        ref_times = [times[i] for i in sung_indices]

    # ===== 收集原始传入数据的调试信息（写入请求日志，便于排查对齐问题） =====
    debug_info = []
    debug_info.append(f"原始文本行数: {len(raw_lines)}")
    debug_info.append(f"演唱歌词行数: {len(sung_lines)}")
    debug_info.append(f"参考时间戳数量: {len(ref_times) if ref_times else 0}")
    debug_info.append("-" * 60)
    for idx, line in enumerate(sung_lines):
        t = f"{ref_times[idx]:8.2f}s" if ref_times and idx < len(ref_times) else "  无时间  "
        debug_info.append(f"  [{idx:3d}] {t} | {line}")
    debug_info_str = "\n".join(debug_info)

    all_words = []

    def fix_stacked_timestamps(words):
        """
        检测并修复因对齐失败导致的时间戳堆叠问题。

        当 stable_whisper 对齐失败时，失败的词会被赋予相同的时间戳（通常是段尾时间），
        导致大量词堆积在同一时间点。此函数检测连续多个词时间戳完全相同的"卡住"区域，
        并通过线性插值修复，最后做单调递增校正保证时间戳不倒退。
        """
        if len(words) < 2:
            return words

        fixed = [SimpleNamespace(word=w.word, start=w.start, end=w.end) for w in words]
        n = len(fixed)

        # 检测"卡住"区域：连续多个词挤在极短的时间跨度内（间隔极小）。
        # 对齐失败时，stable_whisper 可能给所有词相同时间戳，也可能给间隔极小（0.01s）
        # 的时间戳。因此用"时间跨度"而非"完全相同"来检测，更稳健。
        MIN_STUCK_RUN = 15   # 至少连续15个词挤在一起才视为卡住（正常歌词一行最多10-15字）
        STUCK_SPAN = 0.5     # 挤在一起的时间跨度阈值（秒）

        i = 0
        while i < n:
            # 从 i 开始，找到连续挤在一起的区域（时间跨度 < STUCK_SPAN）
            j = i + 1
            while j < n and (fixed[j].start - fixed[i].start) < STUCK_SPAN:
                j += 1

            run_len = j - i
            if run_len >= MIN_STUCK_RUN:
                # 找到卡住区域，进行插值修复
                # 前边界：卡住区域前最后一个词的 end 时间
                prev_end = fixed[i - 1].end if i > 0 else 0.0

                # 后边界：卡住区域后第一个词的 start 时间
                next_start = fixed[j].start if j < n else None

                # 确定插值终点：优先用后一个正常词的 start，否则向后展开
                if next_start is not None and next_start > prev_end:
                    end_t = next_start
                else:
                    # 卡住区域在末尾，或后边界不合法：向后展开，每词至少 0.3s
                    end_t = prev_end + max(run_len * 0.3, 1.0)

                total_time = end_t - prev_end
                step = total_time / (run_len + 1)

                for k in range(run_len):
                    fixed[i + k].start = prev_end + step * (k + 1)
                    fixed[i + k].end = prev_end + step * (k + 1) + step * 0.8

                i = j
            else:
                i += 1

        # ===== 单调递增校正：确保整个序列时间戳不倒退 =====
        # 插值可能产生时间倒退（卡住区域前有正常词，但插值起点早于这些词），
        # 这里把每个词的 start 推到前一个词的 end 之后，保证时间单调递增。
        for k in range(1, n):
            if fixed[k].start < fixed[k - 1].end:
                fixed[k].start = fixed[k - 1].end
                fixed[k].end = max(fixed[k].end, fixed[k].start + 0.01)

        return fixed

    # 参考时间戳有效：长度匹配且存在非零时间（避免无时间戳歌词误入分段逻辑）
    has_valid_times = ref_times and len(ref_times) == len(sung_lines) and max(ref_times) > 0

    if has_valid_times:
        # ===== 参考时间戳可靠性检测：检测"堆叠"区域 =====
        # 网易云标准 LRC 时间戳在歌曲后半部分可能完全失效：大量连续行被错误地
        # 堆叠在同一个时间点（间隔仅 0.01-0.1s）。此时参考时间戳已不可靠，需要改用
        # "切片 + 整段对齐"（让模型自己在切片范围内识别歌词时间）。
        # 注意：这里不做"跳跃校正"（把大间隔误判为异常跳跃并前移时间戳），因为
        # 大间隔往往是真实的间奏（如副歌之间的间奏），前移会破坏正常的时间戳，
        # 导致切片范围错误、段末歌词对齐失败。
        SEGMENT_GAP = 10.0  # 相邻行时间差超过该值视为间奏断点（秒）
        STACK_RUN = 5      # 连续至少 5 行挤在一起视为堆叠
        STACK_SPAN = 2.0   # 挤在一起的时间跨度阈值（秒）
        stack_start = None
        i = 0
        while i < len(ref_times):
            j = i + 1
            while j < len(ref_times) and (ref_times[j] - ref_times[i]) < STACK_SPAN:
                j += 1
            if (j - i) >= STACK_RUN:
                stack_start = i
                break
            i += 1

        if stack_start is not None:
            print(f"⚠️ 参考时间戳在第 {stack_start} 行之后出现堆叠（{j-i} 行挤在 "
                  f"{ref_times[j-1]-ref_times[i]:.2f}s 内），参考时间戳后半部分不可靠！")
            # 后半部分包含堆叠前 2 行作为锚点，帮助模型定位切片起始位置
            ANCHOR_LINES = 2
            anchor_start = max(0, stack_start - ANCHOR_LINES)
            print(f"🔄 改用「切片 + 整段对齐」策略：前半(1-{anchor_start}行)用参考时间戳，"
                  f"后半({anchor_start+1}-{len(sung_lines)}行，含{ANCHOR_LINES}行锚点)让模型自行识别。")
            # ===== 方案：前半用参考时间戳分段对齐，后半用切片 + 整段对齐 =====
            audio = load_audio(audio_path)
            audio_duration = audio.shape[0] / 16000.0

            # 前半部分：堆叠区域之前的歌词（索引 0 到 anchor_start-1）
            # 用参考时间戳分段对齐（与现有逻辑相同）
            if anchor_start > 0:
                front_segments = []
                seg_start = 0
                for k in range(1, anchor_start):
                    if ref_times[k] - ref_times[k-1] > SEGMENT_GAP:
                        front_segments.append((seg_start, k-1))
                        seg_start = k
                front_segments.append((seg_start, anchor_start-1))
                # 合并单行段落
                merged = []
                for (s, e) in front_segments:
                    if merged and (e - s + 1) == 1:
                        prev_s, prev_e = merged[-1]
                        merged[-1] = (prev_s, e)
                    else:
                        merged.append((s, e))
                front_segments = merged

                print(f"🧠 前半部分（第 1-{anchor_start} 行）用参考时间戳分段对齐，共 {len(front_segments)} 段...")
                for (s, e) in front_segments:
                    seg_text = "\n".join(sung_lines[s:e+1])
                    PAD_BEFORE = 1.0
                    PAD_AFTER = 5.0
                    start_t = max(0.0, ref_times[s] - PAD_BEFORE)
                    seg_tail_min = ref_times[e] + 5.0 + PAD_AFTER
                    if e + 1 < anchor_start:
                        end_t = max(ref_times[e+1] + PAD_AFTER, seg_tail_min)
                    else:
                        # 前半最后一段：延伸到锚点区域起点 + 余量
                        end_t = max(ref_times[anchor_start] + PAD_AFTER, seg_tail_min)
                    start_sample = int(start_t * 16000)
                    end_sample = min(int(end_t * 16000), audio.shape[0])
                    if end_sample <= start_sample:
                        continue
                    seg_audio = audio[start_sample:end_sample]
                    print(f"  📄 段落 {s+1}-{e+1}: [{start_t:.2f}s - {end_t:.2f}s]")
                    seg_result = model.align(seg_audio, seg_text, language=lang)
                    if seg_result is None:
                        continue
                    for seg in seg_result.segments:
                        for w in seg.words:
                            w.start += start_t
                            w.end += start_t
                            all_words.append(w)

            # 后半部分：堆叠区域及之后的歌词
            # 关键：包含堆叠前 2 行作为锚点（锚点歌词时间戳可靠，帮助模型定位切片起始位置）
            ANCHOR_LINES = 2
            anchor_start = max(0, stack_start - ANCHOR_LINES)
            tail_text = "\n".join(sung_lines[anchor_start:])
            # 切片起点：锚点行的参考时间 - 余量
            if anchor_start > 0:
                tail_start = max(0.0, ref_times[anchor_start-1] - 2.0)
            else:
                tail_start = 0.0
            tail_end = audio_duration
            tail_audio = audio[int(tail_start*16000):int(tail_end*16000)]
            print(f"🧠 后半部分（第 {anchor_start+1}-{len(sung_lines)} 行，含 {ANCHOR_LINES} 行锚点）"
                  f"用切片 + 整段对齐，切片 [{tail_start:.2f}s - {tail_end:.2f}s]...")
            tail_result = model.align(tail_audio, tail_text, language=lang)
            if tail_result is not None:
                for seg in tail_result.segments:
                    for w in seg.words:
                        w.start += tail_start
                        w.end += tail_start
                        all_words.append(w)

            # ===== 修复时间戳堆叠：对齐失败的词会被赋予相同时间戳，需要插值修复 =====
            all_words = fix_stacked_timestamps(all_words)
        else:
            # ===== 参考时间戳可靠，先找最大间奏断点，再决定是否分段 =====
            # 找最大间奏间隔作为断点后，根据歌词长度和间奏大小决定策略：
            # - 短歌词（≤25行）或间奏小（≤8s）：整段对齐（效果更好）
            # - 长歌词（>25行）+ 大间奏（>8s）：2 段方案（避免模型漂移）
            # 2 段方案细节：段1 切片精确（0 到断点前参考时间），
            # 段2 切片延伸到音频末尾（模型有足够范围识别所有歌词）。
            audio = load_audio(audio_path)  # 16kHz numpy 数组
            audio_duration = audio.shape[0] / 16000.0

            # ===== 参考时间戳保护：若参考时间远超音频实际长度，按比例缩放 =====
            # 网易云标准 LRC 时间可能比实际发音晚很多，甚至超过音频末尾。
            # 若不缩放，切片会超出音频范围，导致对齐结果全部堆叠在同一时间点。
            if ref_times[-1] > audio_duration:
                scale = audio_duration / ref_times[-1]
                print(f"⚠️ 参考时间戳末尾 {ref_times[-1]:.2f}s 超过音频实际长度 {audio_duration:.2f}s，"
                      f"按比例缩放 ref_times (×{scale:.3f})")
                ref_times = [t * scale for t in ref_times]

            # 找最大间奏间隔作为断点
            max_gap = 0
            break_idx = -1
            for i in range(1, len(ref_times)):
                gap = ref_times[i] - ref_times[i-1]
                if gap > max_gap:
                    max_gap = gap
                    break_idx = i  # 断点在第 i 行之前

            # ===== 决策：整段对齐 vs 2 段方案 =====
            # 实测短歌词（≤25行）或短音频（<90s）整段处理效果更好，强制分段反而引入边界对齐误差。
            # 2 段方案仅适用于长歌词（>25行）+ 长音频（≥90s）+ 大间奏（>8s）的情况。
            SHORT_LYRICS_THRESHOLD = 25
            SHORT_AUDIO_THRESHOLD = 90.0  # 秒（1分30秒）
            MAX_GAP_THRESHOLD = 8.0
            use_single_segment = (
                len(sung_lines) < 2
                or break_idx < 0
                or len(sung_lines) <= SHORT_LYRICS_THRESHOLD
                or audio_duration < SHORT_AUDIO_THRESHOLD
                or max_gap <= MAX_GAP_THRESHOLD
            )

            if use_single_segment:
                reason = []
                if len(sung_lines) < 2:
                    reason.append(f"只有 {len(sung_lines)} 行歌词")
                elif len(sung_lines) <= SHORT_LYRICS_THRESHOLD:
                    reason.append(f"歌词较短（{len(sung_lines)} 行 ≤ {SHORT_LYRICS_THRESHOLD}）")
                elif audio_duration < SHORT_AUDIO_THRESHOLD:
                    reason.append(f"音频较短（{audio_duration:.1f}s < {SHORT_AUDIO_THRESHOLD:.0f}s）")
                elif max_gap <= MAX_GAP_THRESHOLD:
                    reason.append(f"最大间奏仅 {max_gap:.2f}s ≤ {MAX_GAP_THRESHOLD}s")
                print(f"🧠 {' + '.join(reason)}，跳过 2 段切分，整段对齐...")
                seg_start = max(0.0, ref_times[0] - 5.0) if ref_times else 0.0
                seg_audio = audio[int(seg_start * 16000):]
                result = model.align(seg_audio, full_text, language=lang)
                if result is not None:
                    for seg in result.segments:
                        for w in seg.words:
                            w.start += seg_start
                            w.end += seg_start
                            all_words.append(w)
                all_words = fix_stacked_timestamps(all_words)
            else:
                print(f"🧠 音频 {audio_duration:.1f}s（≥{SHORT_AUDIO_THRESHOLD:.0f}s），"
                      f"歌词 {len(sung_lines)} 行（>{SHORT_LYRICS_THRESHOLD}），"
                      f"最大间奏 {max_gap:.2f}s（>{MAX_GAP_THRESHOLD}s），"
                      f"断点在第 {break_idx} 行之前，用 2 段方案对齐...")

                # 段1：断点前歌词，切片精确
                seg1_lines = sung_lines[:break_idx]
                # 段1起点：从第一个参考时间前留余量，而非固定从 0s 开始。
                PAD_BEFORE = 5.0
                seg1_start = max(0.0, ref_times[0] - PAD_BEFORE)
                # 段1切片终点：在段2起点之前留固定间隔（5s），确保不重叠。
                seg2_start = max(0.0, ref_times[break_idx] - 5.0)
                seg1_end = max(ref_times[break_idx-1] + 5.0, seg2_start - 5.0)
                seg1_audio = audio[int(seg1_start*16000):int(seg1_end*16000)]
                print(f"  📄 段1（{len(seg1_lines)} 行）: [{seg1_start:.2f}s - {seg1_end:.2f}s]")
                seg1_result = model.align(seg1_audio, "\n".join(seg1_lines), language=lang)
                if seg1_result is not None:
                    for seg in seg1_result.segments:
                        for w in seg.words:
                            w.start += seg1_start
                            w.end += seg1_start
                            all_words.append(w)

                # 段2：断点后歌词，切片延伸到音频末尾
                seg2_lines = sung_lines[break_idx:]
                seg2_end = audio_duration
                seg2_audio = audio[int(seg2_start*16000):int(seg2_end*16000)]
                print(f"  📄 段2（{len(seg2_lines)} 行）: [{seg2_start:.2f}s - {seg2_end:.2f}s]")
                seg2_result = model.align(seg2_audio, "\n".join(seg2_lines), language=lang)
                if seg2_result is not None:
                    for seg in seg2_result.segments:
                        for w in seg.words:
                            w.start += seg2_start
                            w.end += seg2_start
                            all_words.append(w)

                # ===== 修复时间戳堆叠：对齐失败的词会被赋予相同时间戳，需要插值修复 =====
                all_words = fix_stacked_timestamps(all_words)
    else:
        # ===== 无参考时间戳，整段对齐 =====
        print("🧠 正在进行强制对齐推理...")
        result = model.align(audio_path, full_text, language=lang)
        for seg in result.segments:
            for w in seg.words:
                all_words.append(w)

    lrc_lines = []
    enhanced_lrc_lines = []
    
    meta_info = []
    if ti: meta_info.append(f"[ti:{ti}]")
    if ar: meta_info.append(f"[ar:{ar}]")
    if al: meta_info.append(f"[al:{al}]")
    lrc_lines.extend(meta_info)
    enhanced_lrc_lines.extend(meta_info)
    
    if staff_lines:
        first_word_start = all_words[0].start if all_words else 0.0
        safe_intro = max(0, first_word_start - 0.5)
        interval = safe_intro / max(1, len(staff_lines))
        for i, staff in enumerate(staff_lines):
            staff_str = f"{format_time(i * interval)}{staff}"
            lrc_lines.append(staff_str)
            enhanced_lrc_lines.append(staff_str)

    word_idx = 0
    total_words = len(all_words)
    prev_line_end_time = -1.0
    interlude_threshold = 3.0 

    for line in sung_lines:
        if word_idx < total_words:
            start_time = all_words[word_idx].start
            start_time_str = format_time(start_time)
        else:
            start_time = prev_line_end_time + 0.1
            start_time_str = "[99:99.99]" 

        if prev_line_end_time > 0 and (start_time - prev_line_end_time) > interlude_threshold:
            interlude_str = f"{format_time(prev_line_end_time + 0.2)} "
            lrc_lines.append(interlude_str)
            enhanced_lrc_lines.append(interlude_str)

        # 按原始歌词拆分为匹配段：CJK 逐字，拉丁文逐词（保留单词边界和位置信息）
        original_words, seg_ends = split_line_to_segments(line)
        line_word_results = []  # [(orig_word, start, end, seg_end_pos), ...]

        for seg_idx, orig_word in enumerate(original_words):
            target = clean_str(orig_word)
            if not target:
                continue

            accumulated = ""
            w_start = None
            w_end = None

            while word_idx < total_words and len(accumulated) < len(target):
                current_word = all_words[word_idx]
                word_text = clean_str(current_word.word)

                if not word_text:
                    word_idx += 1
                    continue

                new_acc = accumulated + word_text

                if target.startswith(new_acc):
                    # 模型词完全在当前原始单词范围内
                    if w_start is None:
                        w_start = current_word.start
                    w_end = current_word.end
                    accumulated = new_acc
                    word_idx += 1
                elif len(word_text) > (len(target) - len(accumulated)):
                    # 模型词可能跨越原始单词边界，尝试拆分
                    remaining = len(target) - len(accumulated)
                    if remaining > 0 and word_text[:remaining] == target[len(accumulated):]:
                        if w_start is None:
                            w_start = current_word.start
                        tail_text = word_text[remaining:]
                        ratio = remaining / len(word_text)
                        head_end = current_word.start + (current_word.end - current_word.start) * ratio
                        w_end = head_end
                        accumulated = target
                        if tail_text:
                            tail_word = SimpleNamespace(word=tail_text, start=head_end, end=current_word.end)
                            all_words[word_idx] = tail_word
                        else:
                            word_idx += 1
                    else:
                        # 模型词不匹配当前原始单词（对齐错位），跳过
                        word_idx += 1
                        continue
                else:
                    # 模型词不属于当前原始单词（对齐错位），跳过
                    word_idx += 1
                    continue

            if w_start is not None and accumulated:
                line_word_results.append((orig_word, w_start, w_end, seg_ends[seg_idx]))

        # 标准 LRC：用行首词的 start 时间
        if line_word_results:
            start_time = line_word_results[0][1]
            start_time_str = format_time(start_time)
            current_line_end_time = line_word_results[-1][2]
        else:
            current_line_end_time = prev_line_end_time + 0.1 if prev_line_end_time > 0 else 0.1
            start_time_str = "[99:99.99]"

        lrc_lines.append(f"{start_time_str}{line}")

        # 增强 LRC：使用原始歌词单词，格式 [start]word1 [end1]word2 [end2]...wordN[endN]
        if line_word_results:
            enhanced_line_str = format_time(line_word_results[0][1])
            for i, (word, ws, we, seg_end) in enumerate(line_word_results):
                enhanced_line_str += word
                if i < len(line_word_results) - 1:
                    next_ws = line_word_results[i + 1][1]
                    # 根据原始行中该段之后是否有空格来决定输出空格
                    # 这样既保留了原始歌词的分词（如"爱 动"之间的空格），
                    # 又不会在连续 CJK 字符间插入多余空格（如"天下"之间）
                    has_space_after = seg_end < len(line) and line[seg_end] in (' ', '\u3000')
                    if has_space_after:
                        enhanced_line_str += " " + format_time(we)
                    else:
                        enhanced_line_str += format_time(we)
                    # 若下一词起始与当前词结束不一致（句中停顿），额外标记下一词起始时间
                    if next_ws - we > GAP_THRESHOLD:
                        enhanced_line_str += format_time(next_ws)
                else:
                    enhanced_line_str += format_time(we)
            enhanced_lrc_lines.append(enhanced_line_str)
        else:
            enhanced_lrc_lines.append(f"{start_time_str}{line}")

        prev_line_end_time = current_line_end_time

    if prev_line_end_time > 0:
        end_str = f"{format_time(prev_line_end_time + 1.0)} "
        lrc_lines.append(end_str)
        enhanced_lrc_lines.append(end_str)

    return {
        "standard_lrc": "\n".join(lrc_lines),
        "enhanced_lrc": "\n".join(enhanced_lrc_lines),
        "debug_info": debug_info_str
    }

@app.get("/api/ping")
async def ping():
    return {"status": "ok", "app": "lrc-maker-ai", "version": "2.0"}

@app.post("/api/align")
async def api_align(
    audio: UploadFile = File(...),
    lyrics: str = Form(...),
    ti: str = Form(""),
    ar: str = Form(""),
    al: str = Form(""),
    times: str = Form("")
):
    global _current_request_id

    # 分配本次请求 ID，并标记为当前最新请求
    request_id = next(_next_request_id)
    _current_request_id = request_id

    print(f"📥 收到请求 #{request_id}：音频文件 [{audio.filename}], 文本长度 [{len(lyrics)}]")
    
    tmp_path = None 
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name
            content = await audio.read()
            tmp.write(content)
            
        print(f"🎵 请求 #{request_id} 音频已保存至临时目录，开始处理...")

        # ===== 保存音频副本到项目 audio/ 目录，方便试听/调试 =====
        if SAVE_AUDIO:
            try:
                audio_dir = os.path.join(application_path, "audio")
                os.makedirs(audio_dir, exist_ok=True)
                ext = detect_audio_ext(content)
                base_name = os.path.splitext(audio.filename)[0] if audio.filename else f"audio_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                save_name = f"{base_name}{ext if ext else '.bin'}"
                save_path = os.path.join(audio_dir, save_name)
                with open(save_path, "wb") as f:
                    f.write(content)
                print(f"💾 音频已保存到: {save_path}（可打开试听）")
            except Exception as e:
                print(f"⚠️ 保存音频副本失败: {e}")
        # =========================================================
        
        # 解析参考时间戳（可选，JSON 数组，秒）
        ref_times = None
        if times:
            try:
                ref_times = json.loads(times)
            except Exception:
                ref_times = None

        # 用 asyncio.to_thread 将同步阻塞的模型推理放到线程池执行，
        # 不再卡死事件循环，让新请求可以立即开始处理
        lrc_result_dict = await asyncio.to_thread(
            generate_lrc_content, tmp_path, lyrics, ti, ar, al, times=ref_times
        )

        # 处理完后检查：是否有更新的请求已经进来了？
        # 如果有，本次结果已过时，直接丢弃（避免旧结果覆盖新结果）
        if _current_request_id != request_id:
            print(f"⚠️ 请求 #{request_id} 已过时（当前最新为 #{_current_request_id}），丢弃结果。")
            return {"code": 409, "message": "superseded", "data": None}
        
        if isinstance(lrc_result_dict, dict) and "error" in lrc_result_dict:
            print(lrc_result_dict["error"])
            return {"code": 400, "message": lrc_result_dict["error"], "data": None}
            
        log_request(audio.filename, lyrics, lrc_result_dict)
        
        lrc_result_dict.pop("debug_info", None)
        
        print(f"✅ 请求 #{request_id} 处理完成，返回标准与逐字双轨数据给前端。")
        return {"code": 200, "message": "success", "data": lrc_result_dict}
        
    except Exception as e:
        print(f"❌ 请求 #{request_id} 发生错误: {str(e)}")
        return {"code": 500, "message": str(e), "data": None}
        
    finally:
        if tmp_path and os.path.exists(tmp_path):
            for _ in range(5):
                try:
                    os.unlink(tmp_path)
                    break
                except PermissionError:
                    import time
                    time.sleep(0.2)
                except FileNotFoundError:
                    break
                except Exception as e:
                    print(f"⚠️ 删除临时文件失败（将忽略）: {e}")
                    break

def find_free_port(start_port=8000):
    """从指定的起始端口开始，寻找一个未被占用的本地端口"""
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
        port += 1
    raise RuntimeError("无法找到可用的端口！")

if __name__ == "__main__":
    multiprocessing.freeze_support()

    active_port = find_free_port(8000)
    
    print("\n" + "="*50)
    print("🚀 LRCMaker 本地后端服务已点火！")
    print(f"🔌 当前 API 监听端口: {active_port}")
    if active_port != 8000:
        print(f"⚠️  注意：默认端口 8000 已被占用，已自动切换至 {active_port}")
        print(f"👉  【重要】请在你的浏览器插件/前端配置中，将后端地址改为: http://127.0.0.1:{active_port}")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="127.0.0.1", port=active_port)