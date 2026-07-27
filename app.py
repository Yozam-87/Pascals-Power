import os
import json
import subprocess
import shlex
import struct
import sys
import shutil
from dotenv import load_dotenv

# =====================================================================
# INITIALIZATION & CONFIGURATION
# =====================================================================
# Load the .env file variables into the system environment
load_dotenv()

# Pull from .env, with safe fallbacks for open-source users
# Prefer PATH lookup, fall back to .env / hardcoded default
_llama_resolved = shutil.which("llama-server")
LLAMA_SERVER_PATH = os.getenv("LLAMA_SERVER_PATH", _llama_resolved or "./llama-server")
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
WEB_PORT = int(os.getenv("PORT", 5005))
ENV_VARS_RAW = os.getenv("ENV_VARS", "")

def parse_env_vars(raw: str) -> dict:
    """Parse pipe-separated KEY=VALUE pairs into a dict.
    
    Example: 'CUDA_VISIBLE_DEVICES=0|NCCL_DEBUG=INFO'
    """
    env = {}
    if not raw or not raw.strip():
        return env
    for pair in raw.split('|'):
        pair = pair.strip()
        if '=' in pair:
            key, _, value = pair.partition('=')
            key = key.strip()
            value = value.strip()
            if key:
                env[key] = value
    return env

ENV_VARS = parse_env_vars(ENV_VARS_RAW)

# Auto-create the models directory if it doesn't exist 
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR, exist_ok=True)

# Define static workspace paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_FILE = os.path.join(BASE_DIR, "profiles.json")
LOG_FILE = os.path.join(BASE_DIR, "llama_server.log")

template_dir = os.path.join(BASE_DIR, "templates")
if not os.path.exists(template_dir):
    template_dir = BASE_DIR

# =====================================================================
# SYSTEM RESOURCE LIMITS (Disables SSD-killing core dumps on crash)
# =====================================================================
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    print("🔒 Core dumps globally disabled (RLIMIT_CORE set to 0 to protect SSD).")
except Exception as e:
    print(f"⚠️ Warning: Could not disable core dumps: {e}", file=sys.stderr)

from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder=template_dir)
current_process = None
current_log_file = None
current_profile = None

# Helper to automatically strip wrapping quotes (Subprocess bypasses the shell)
def strip_quotes(s):
    if not s: return ""
    s = s.strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1].strip()
    return s

def resolve_path(p):
    if not p: return ""
    p = strip_quotes(p)
    if p.startswith("/") or p.startswith("./") or p.startswith("../") or ":" in p: 
        return p
    return os.path.join(MODELS_DIR, p)

# =====================================================================
# PURE GGUF READER
# =====================================================================
class PureGGUFReader:
    def __init__(self, filepath):
        self.f = open(filepath, 'rb')
        self.metadata = {}
        self.tensors = []
        self._parse()
        self.f.close()

    def get_metadata_val(self, suffix):
        for k, v in self.metadata.items():
            if k.endswith(suffix): return v
        return None

    def _read_str(self):
        length = struct.unpack('<Q', self.f.read(8))[0]
        return self.f.read(length).decode('utf-8', errors='ignore')

    def _read_val(self, val_type):
        if val_type == 0: return struct.unpack('<B', self.f.read(1))[0]
        elif val_type == 1: return struct.unpack('<b', self.f.read(1))[0]
        elif val_type == 2: return struct.unpack('<H', self.f.read(2))[0]
        elif val_type == 3: return struct.unpack('<h', self.f.read(2))[0]
        elif val_type == 4: return struct.unpack('<I', self.f.read(4))[0]
        elif val_type == 5: return struct.unpack('<i', self.f.read(4))[0]
        elif val_type == 6: return struct.unpack('<f', self.f.read(4))[0]
        elif val_type == 7: return struct.unpack('<?', self.f.read(1))[0]
        elif val_type == 8: return self._read_str()
        elif val_type == 9:
            arr_type = struct.unpack('<I', self.f.read(4))[0]
            arr_len = struct.unpack('<Q', self.f.read(8))[0]
            item_sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
            if arr_type in item_sizes:
                raw_data = self.f.read(item_sizes[arr_type] * arr_len)
                try:
                    fmts = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}
                    return list(struct.unpack(f"<{arr_len}{fmts[arr_type]}", raw_data))
                except Exception: return f"[Array of {arr_len} items]"
            elif arr_type == 8:
                return [self.f.read(struct.unpack('<Q', self.f.read(8))[0]).decode('utf-8', errors='ignore') for _ in range(arr_len)]
            return "[Array of unknown type]"
        elif val_type == 10: return struct.unpack('<Q', self.f.read(8))[0]
        elif val_type == 11: return struct.unpack('<q', self.f.read(8))[0]
        elif val_type == 12: return struct.unpack('<d', self.f.read(8))[0]
        return None

    def _parse(self):
        if self.f.read(4) != b'GGUF': raise ValueError("Not a valid GGUF file")
        struct.unpack('<I', self.f.read(4))[0]
        tensor_count = struct.unpack('<Q', self.f.read(8))[0]
        kv_count = struct.unpack('<Q', self.f.read(8))[0]

        for _ in range(kv_count):
            key = self._read_str()
            self.metadata[key] = self._read_val(struct.unpack('<I', self.f.read(4))[0])

        type_bytes = {0:4.0, 1:2.0, 2:0.5, 3:0.5, 6:1.0, 7:0.5625, 8:0.5625, 9:0.25, 10:0.375, 11:0.5, 12:0.5625, 13:0.75, 14:1.0, 15:0.25, 16:0.275, 17:0.375, 18:0.1875, 19:0.5625, 20:0.43, 21:0.3125, 22:0.53, 28:0.22}
        for _ in range(tensor_count):
            name = self._read_str()
            ndim = struct.unpack('<I', self.f.read(4))[0]
            dims = [struct.unpack('<Q', self.f.read(8))[0] for _ in range(ndim)]
            tensor_type = struct.unpack('<I', self.f.read(4))[0]
            struct.unpack('<Q', self.f.read(8))[0]
            elem_count = 1
            for d in dims: elem_count *= d
            self.tensors.append({'name': name, 'n_bytes': elem_count * type_bytes.get(tensor_type, 2.0)})

# =====================================================================
# API ROUTES
# =====================================================================
if not os.path.exists(PROFILES_FILE):
    with open(PROFILES_FILE, "w") as f: json.dump({"default": {"model": "", "flags": {"--port": "5001"}}}, f)

def load_profiles():
    with open(PROFILES_FILE, "r") as f: return json.load(f)

def save_profiles(profiles):
    with open(PROFILES_FILE, "w") as f: json.dump(profiles, f, indent=4)

@app.route("/")
def index(): 
    return render_template("index.html", models_dir=MODELS_DIR)

@app.route("/api/config", methods=["GET"])
def get_config():
    """Return server configuration for the frontend (command preview, etc.)."""
    return jsonify({
        "serverPath": LLAMA_SERVER_PATH,
        "modelsDir": MODELS_DIR,
        "envVars": ENV_VARS or None,
        "currentProfile": current_profile or None,
        "port": WEB_PORT
    })

@app.route("/api/models", methods=["GET"])
def get_models():
    if not os.path.exists(MODELS_DIR):
        return jsonify([])
    files = [f for f in os.listdir(MODELS_DIR) if f.lower().endswith('.gguf')]
    return jsonify(sorted(files))

@app.route("/api/profiles", methods=["GET"])
def get_profiles(): return jsonify(load_profiles())

@app.route("/api/profiles/<name>", methods=["POST"])
def save_profile(name):
    profiles = load_profiles()
    profiles[name] = request.json
    save_profiles(profiles)
    return jsonify({"status": "saved"})

@app.route("/api/profiles/<name>", methods=["DELETE"])
def delete_profile(name):
    profiles = load_profiles()
    if name in profiles:
        del profiles[name]
        save_profiles(profiles)
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Profile not found"}), 404

@app.route("/api/hardware", methods=["GET", "POST"])
def manage_hardware():
    profiles = load_profiles()
    if request.method == "POST":
        profiles["__hardware__"] = request.json
        save_profiles(profiles)
        return jsonify({"status": "saved"})
    return jsonify(profiles.get("__hardware__", {
        "gpu0_vram": 6.0, "gpu0_overhead": 0.0, "gpu0_safety": 500.0,
        "gpu1_vram": 8.0, "gpu1_overhead": 1024.0, "gpu1_safety": 500.0
    }))

@app.route("/api/analyze", methods=["POST"])
def analyze_model():
    data = request.json
    gguf_path = resolve_path(data.get("model", ""))
    mmproj_path = resolve_path(data.get("mmproj", ""))
    context_length = int(data.get("ctx_size", 8192))
    kv_type = str(data.get("kv_type", "q4_0")).lower()

    if not os.path.exists(gguf_path):
        return jsonify({"error": f"Model file not found: {gguf_path}"}), 404

    profiles_data = load_profiles()
    hw = profiles_data.get("__hardware__", {
        "gpu0_vram": 6.0, "gpu0_overhead": 0.0, "gpu0_safety": 500.0,
        "gpu1_vram": 8.0, "gpu1_overhead": 1024.0, "gpu1_safety": 500.0
    })

    gpu0_vram_gb = float(hw.get("gpu0_vram", 6.0))
    gpu0_overhead_mb = float(hw.get("gpu0_overhead", 0.0))
    gpu0_safety_mb = float(hw.get("gpu0_safety", 500.0))

    gpu1_vram_gb = float(hw.get("gpu1_vram", 8.0))
    gpu1_overhead_mb = float(hw.get("gpu1_overhead", 1024.0))
    gpu1_safety_mb = float(hw.get("gpu1_safety", 500.0))

    try:
        reader = PureGGUFReader(gguf_path)
        block_count = reader.get_metadata_val(".block_count")
        n_embd = reader.get_metadata_val(".embedding_length")
        n_head = reader.get_metadata_val(".attention.head_count")
        n_kv_heads = reader.get_metadata_val(".attention.head_count_kv")
        n_key_length = reader.get_metadata_val(".attention.key_length")
        n_val_length = reader.get_metadata_val(".attention.value_length")

        full_attn_interval = reader.get_metadata_val(".full_attention_interval")
        ssm_state_size = reader.get_metadata_val(".ssm.state_size")
        ssm_inner_size = reader.get_metadata_val(".ssm.inner_size")
        sliding_window = reader.get_metadata_val(".attention.sliding_window")

        if isinstance(block_count, list): total_layers = len(block_count)
        else: total_layers = int(block_count) if block_count is not None else 32

        if isinstance(n_embd, list): n_embd = sum(n_embd) / len(n_embd)
        else: n_embd = int(n_embd) if n_embd is not None else 4096

        if isinstance(n_head, list): n_head = sum(n_head) / len(n_head)
        else: n_head = int(n_head) if n_head is not None else 32

        if isinstance(n_kv_heads, list): n_kv_heads = sum(n_kv_heads) / len(n_kv_heads)
        else: n_kv_heads = int(n_kv_heads) if n_kv_heads is not None else (n_head // 8 if n_head else 8)

        if isinstance(n_key_length, list): head_dim = sum(n_key_length) / len(n_key_length)
        elif isinstance(n_val_length, list): head_dim = sum(n_val_length) / len(n_val_length)
        elif n_key_length is not None: head_dim = int(n_key_length)
        elif n_val_length is not None: head_dim = int(n_val_length)
        else: head_dim = n_embd // n_head if n_head else 128

        is_hybrid = full_attn_interval is not None or ssm_state_size is not None
        swa_window = int(sliding_window) if sliding_window is not None else None

        base_overhead = 0
        layer_tensors = {}
        expert_tensors = {}

        for tensor in reader.tensors:
            name = tensor['name']
            n_bytes = tensor['n_bytes']
            if name.startswith("blk."):
                parts = name.split(".")
                try: layer_idx = int(parts[1])
                except ValueError: continue
                is_expert = "exps" in name
                if is_expert: expert_tensors[layer_idx] = expert_tensors.get(layer_idx, 0) + n_bytes
                else: layer_tensors[layer_idx] = layer_tensors.get(layer_idx, 0) + n_bytes
            else: base_overhead += n_bytes

        avg_layer_mb = (sum(layer_tensors.values()) / len(layer_tensors)) / (1024**2) if layer_tensors else 140.0
        is_moe = len(expert_tensors) > 0
        avg_expert_mb = (sum(expert_tensors.values()) / len(expert_tensors)) / (1024**2) if is_moe else 0.0

        bytes_per_elem = 2.0 if kv_type == "f16" else (1.0 if kv_type == "q8_0" else 0.5)
        kv_bytes_per_layer = 2 * n_kv_heads * head_dim * context_length * bytes_per_elem
        kv_mb_per_layer = kv_bytes_per_layer / (1024**2)

        if swa_window is not None:
            arch = str(reader.get_metadata_val("general.architecture") or "").lower()
            global_ratio = 0.5 if "gemma2" in arch else (1.0 / 6.0)
            global_layers = max(1, int(total_layers * global_ratio))
            local_layers = total_layers - global_layers
            kv_bytes_local_layer = 2 * n_kv_heads * head_dim * swa_window * bytes_per_elem
            kv_mb_local_layer = kv_bytes_local_layer / (1024**2)
            total_cache_cost = (global_layers * kv_mb_per_layer) + (local_layers * kv_mb_local_layer)
        elif is_hybrid:
            interval = int(full_attn_interval) if full_attn_interval is not None else 4
            kv_bearing_layers = total_layers // interval
            deltanet_state_mb = ((int(ssm_state_size) * int(ssm_inner_size) * 4) / (1024**2)) if (ssm_state_size and ssm_inner_size) else 2.0
            total_cache_cost = (kv_bearing_layers * kv_mb_per_layer) + ((total_layers - kv_bearing_layers) * deltanet_state_mb)
        else:
            total_cache_cost = total_layers * kv_mb_per_layer

        graph_overhead_mb = 900.0 + (context_length * 0.001)

        mmproj_size_mb = 0.0
        if mmproj_path and os.path.exists(mmproj_path):
            mmproj_size_mb = os.path.getsize(mmproj_path) / (1024**2)

        usable_gpu0 = (gpu0_vram_gb * 1024) - gpu0_overhead_mb - gpu0_safety_mb
        usable_gpu1 = (gpu1_vram_gb * 1024) - gpu1_overhead_mb - gpu1_safety_mb
        base_overhead_mb = (base_overhead / (1024**2)) + mmproj_size_mb

        if is_moe:
            all_attention_weights = total_layers * avg_layer_mb
            gpu0_used = base_overhead_mb + all_attention_weights + total_cache_cost + graph_overhead_mb
            gpu0_headroom = usable_gpu0 - gpu0_used
            
            print(f"\n[CALC DEBUG] usable_gpu0: {usable_gpu0:.1f}MB, base_overhead: {base_overhead_mb:.1f}MB", file=sys.stderr)
            print(f"[CALC DEBUG] attention_layers: {all_attention_weights:.1f}MB, cache: {total_cache_cost:.1f}MB", file=sys.stderr)
            print(f"[CALC DEBUG] gpu0_headroom: {gpu0_headroom:.1f}MB, expert_size: {avg_expert_mb:.1f}MB", file=sys.stderr)
            
            gpu0_experts = max(0, int(gpu0_headroom // avg_expert_mb)) if gpu0_headroom > avg_expert_mb else 0
            gpu1_experts = min(int(usable_gpu1 // avg_expert_mb), total_layers - gpu0_experts)
            
            return jsonify({
                "is_moe": True,
                "total_layers": total_layers,
                "c0": gpu0_experts,
                "c1": gpu1_experts,
                "ts": "1,0",
                "n_gpu_layers": total_layers
            })
        else:
            layer_cost = avg_layer_mb + (total_cache_cost / total_layers)
            gpu0_for_layers = usable_gpu0 - base_overhead_mb - graph_overhead_mb
            layers_gpu0 = max(0, min(int(gpu0_for_layers // layer_cost), total_layers)) if gpu0_for_layers > 0 else 0
            layers_gpu1 = max(0, min(int(usable_gpu1 // layer_cost), total_layers - layers_gpu0))
            
            return jsonify({
                "is_moe": False,
                "total_layers": total_layers,
                "c0": layers_gpu0,
                "c1": layers_gpu1,
                "ts": f"{layers_gpu0},{layers_gpu1}",
                "n_gpu_layers": layers_gpu0 + layers_gpu1
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/start", methods=["POST"])
def start_server():
    global current_process, current_log_file, current_profile
    data = request.json
    
    cmd = []
    path_parts = shlex.split(LLAMA_SERVER_PATH)
    cmd.extend(path_parts)
    
    if data.get("model"): cmd.extend(["--model", resolve_path(data["model"])])
    if data.get("mmproj"): cmd.extend(["--mmproj", resolve_path(data["mmproj"])])
        
    for key, value in data.get("flags", {}).items():
        value_str = strip_quotes(str(value).strip()) 
        if value is True or value_str.lower() == "true" or value_str == "":
            cmd.append(key)
        elif value is False or value_str.lower() == "false":
            continue
        else:
            cmd.extend([key, value_str])

    if current_process and current_process.poll() is None:
        current_process.terminate()
        current_process.wait()
    if current_log_file:
        current_log_file.close()

    current_profile = data.get("profile", None)
    
    print("\n🚀 LAUNCHING COMMAND:")
    print(" ".join(shlex.quote(arg) for arg in cmd))
    
    current_log_file = open(LOG_FILE, "w")
    # Merge explicit env vars (from .env ENV_VARS) with the current process env
    launch_env = dict(os.environ)
    launch_env.update(ENV_VARS)
    current_process = subprocess.Popen(cmd, stdout=current_log_file, stderr=subprocess.STDOUT, env=launch_env)
    
    return jsonify({"status": "started"})

@app.route("/api/stop", methods=["POST"])
def stop_server():
    global current_process, current_log_file, current_profile
    if current_process and current_process.poll() is None:
        current_process.terminate()
        current_process.wait()
    current_profile = None
    if current_log_file:
        current_log_file.close()
        current_log_file = None
    return jsonify({"status": "stopped"})

@app.route("/api/logs", methods=["GET"])
def get_logs():
    global current_process
    running = current_process is not None and current_process.poll() is None
    
    logs = ""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", errors="ignore") as f:
                lines = f.readlines()
                logs = "".join(lines[-150:])
        except Exception as e:
            logs = f"Error reading logs: {e}"
            
    return jsonify({
        "logs": logs,
        "running": running
    })

if __name__ == "__main__":
    app.run(port=WEB_PORT, debug=False, host="127.0.0.1")