import os
import sys
import json
import socket
import tempfile
import datetime
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
ENABLE_REQUEST_LOG = False
# =============================

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
    else:
        print(f"🌐 未发现本地离线模型，将尝试从网络或系统缓存加载 '{MODEL_SIZE}' 模型...")
        print(f"   (提示：你可以手动下载模型并放入 {local_model_path} 目录实现完全离线运行)")
        model_target = MODEL_SIZE
    
    device = pick_device()
    
    model = stable_whisper.load_faster_whisper(
        model_target, 
        device=device, 
        compute_type="float16" if device == "cuda" else "int8",
        download_root=model_cache_dir
    )
    
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
    return s.replace(" ", "").replace("\u3000", "").replace("\n", "").replace("\r", "").replace("\t", "")

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
    lang = detect_language(full_text)
    print(f"🌐 检测到歌词语言: {lang}")

    # 解析参考时间戳（秒），对应 sung_lines 每行
    ref_times = None
    if times and len(times) == len(raw_lines):
        ref_times = [times[i] for i in sung_indices]

    all_words = []

    # 参考时间戳有效：长度匹配且存在非零时间（避免无时间戳歌词误入分段逻辑）
    has_valid_times = ref_times and len(ref_times) == len(sung_lines) and max(ref_times) > 0

    if has_valid_times:
        # ===== 分段对齐：根据参考时间戳把歌词分段，避免长间奏导致的对齐漂移 =====
        SEGMENT_GAP = 10.0  # 相邻行时间差超过该值视为间奏断点（秒）
        segments = []      # 每段是 (start_idx, end_idx) 闭区间
        seg_start = 0
        for i in range(1, len(sung_lines)):
            if ref_times[i] - ref_times[i-1] > SEGMENT_GAP:
                segments.append((seg_start, i-1))
                seg_start = i
        segments.append((seg_start, len(sung_lines)-1))

        # 合并单行段落：单行歌词对齐效果差，合并到前一段
        merged = []
        for (s, e) in segments:
            if merged and (e - s + 1) == 1:
                prev_s, prev_e = merged[-1]
                merged[-1] = (prev_s, e)
            else:
                merged.append((s, e))
        segments = merged

        print(f"🧠 检测到 {len(segments)} 个歌词段落，进行分段强制对齐...")
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

        for (s, e) in segments:
            seg_text = "\n".join(sung_lines[s:e+1])
            # 音频切片前后各留余量，避免参考时间戳偏差导致歌词被截断
            PAD_BEFORE = 1.0  # 段前余量（秒）
            PAD_AFTER = 3.0   # 段后余量（秒）
            start_t = max(0.0, ref_times[s] - PAD_BEFORE)
            # 段末余量：网易云参考时间可能比实际发音早，若只按下一段参考时间切，
            # 会截断当前段最后一行（"Failed to align the last N words"）。
            # 因此段末至少留 ref_times[e] + 5s 的余量，再与下一段参考时间取较大值。
            seg_tail_min = ref_times[e] + 5.0 + PAD_AFTER
            if e + 1 < len(sung_lines):
                end_t = max(ref_times[e+1] + PAD_AFTER, seg_tail_min)
            else:
                end_t = seg_tail_min
            start_sample = int(start_t * 16000)
            # 限制切片不超出音频末尾，避免空切片/越界导致对齐错乱
            end_sample = min(int(end_t * 16000), audio.shape[0])
            if end_sample <= start_sample:
                print(f"  ⚠️ 段落 {s+1}-{e+1} 切片为空，跳过")
                continue
            seg_audio = audio[start_sample:end_sample]

            print(f"  📄 段落 {s+1}-{e+1}: [{start_t:.2f}s - {end_t:.2f}s]")
            seg_result = model.align(seg_audio, seg_text, language=lang)
            if seg_result is None:
                continue
            for seg in seg_result.segments:
                for w in seg.words:
                    # 时间戳加上段起始偏移（align 返回相对切片开头的时间）
                    w.start += start_t
                    w.end += start_t
                    all_words.append(w)
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

        target = clean_str(line)
        target_len = len(target)
        current_line_end_time = start_time
        
        current_line_words = []
        current_text = ""

        while word_idx < total_words and len(current_text) < target_len:
            current_word = all_words[word_idx]
            word_text = clean_str(current_word.word)
            if not word_text:
                word_idx += 1
                continue

            new_text = current_text + word_text
            if target.startswith(new_text):
                # 整个词都在当前行内
                current_line_words.append(current_word)
                current_text = new_text
                current_line_end_time = current_word.end
                word_idx += 1
            else:
                # 词跨越行边界：当前行取需要的部分，剩余留给下一行
                remaining = target_len - len(current_text)
                if remaining > 0:
                    head_text = word_text[:remaining]
                    tail_text = word_text[remaining:]
                    ratio = remaining / len(word_text)
                    head_end = current_word.start + (current_word.end - current_word.start) * ratio
                    head_word = SimpleNamespace(word=head_text, start=current_word.start, end=head_end)
                    current_line_words.append(head_word)
                    current_text = current_text + head_text
                    current_line_end_time = head_end
                    if tail_text:
                        tail_word = SimpleNamespace(word=tail_text, start=head_end, end=current_word.end)
                        all_words[word_idx] = tail_word
                    else:
                        word_idx += 1
                break

        lrc_lines.append(f"{start_time_str}{line}")
        
        valid_words = [w for w in current_line_words if clean_str(w.word)]
        enhanced_line_str = f"{start_time_str}"
        for i, w in enumerate(valid_words):
            clean_word = clean_str(w.word)
            enhanced_line_str += f"{clean_word}{format_time(w.end)}"
            # 若下一词起始与当前词结束不一致（句中停顿），额外标记下一词起始时间
            if i < len(valid_words) - 1:
                next_w = valid_words[i + 1]
                if next_w.start - w.end > GAP_THRESHOLD:
                    enhanced_line_str += format_time(next_w.start)
        enhanced_lrc_lines.append(enhanced_line_str)

        prev_line_end_time = current_line_end_time

    if prev_line_end_time > 0:
        end_str = f"{format_time(prev_line_end_time + 1.0)} "
        lrc_lines.append(end_str)
        enhanced_lrc_lines.append(end_str)

    return {
        "standard_lrc": "\n".join(lrc_lines),
        "enhanced_lrc": "\n".join(enhanced_lrc_lines)
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
    print(f"📥 收到请求：音频文件 [{audio.filename}], 文本长度 [{len(lyrics)}]")
    
    tmp_path = None 
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name
            content = await audio.read()
            tmp.write(content)
            
        print("🎵 音频已保存至临时目录，开始处理...")
        
        # 解析参考时间戳（可选，JSON 数组，秒）
        ref_times = None
        if times:
            try:
                ref_times = json.loads(times)
            except Exception:
                ref_times = None
        lrc_result_dict = generate_lrc_content(tmp_path, lyrics, ti, ar, al, times=ref_times)
        
        if isinstance(lrc_result_dict, dict) and "error" in lrc_result_dict:
            print(lrc_result_dict["error"])
            return {"code": 400, "message": lrc_result_dict["error"], "data": None}
            
        log_request(audio.filename, lyrics, lrc_result_dict)
        
        print("✅ 处理完成，返回标准与逐字双轨数据给前端。")
        return {"code": 200, "message": "success", "data": lrc_result_dict}
        
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
        return {"code": 500, "message": str(e), "data": None}
        
    finally:
        if tmp_path and os.path.exists(tmp_path):
            # Windows 下音频解码库（soundfile 等）可能未及时释放文件句柄，
            # 直接删除会报 PermissionError: [WinError 32]。这里带重试地删除，
            # 多次失败则忽略（临时文件最终会被系统自动清理）。
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