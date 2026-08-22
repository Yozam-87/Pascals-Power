## 🔴 High

**1.** The `-ot` regex becomes unwieldy for users with high `-ngl` values (e.g. 99). The auto-split calculator should compress consecutive block ranges into shorter regex patterns (e.g. `blk\.([0-9]+)\.` instead of listing each block individually).

**2.** Log output occasionally freezes and stops updating. Toggling auto-scroll sometimes restores it, but recovery is inconsistent. The SSE polling connection to the backend remains stable throughout.

## 🟡 Medium

**3.** The auto-split calculator treats K and V cache types as a single value. The frontend has separate `--cache-type-k` and `--cache-type-v` controls, but only the K value is sent to the backend. Asymmetric K/V configurations produce incorrect VRAM estimates.

**4.** ANSI log colorization is not functional in practice. While the `ansiToHtml()` parser and CSS classes are implemented, `llama-server` strips ANSI escape sequences when stdout is redirected to a file (non-TTY), leaving the frontend colorizer with no data to process.

**5.** When a model name (rather than an absolute path) is entered in the GUI, the generated launch command and copy-button output use a relative path. The server resolves it correctly at runtime, but the copied command is not portable.

## 🟢 Low

**6.** Missing error handling for a missing `llama-server` binary. If the binary cannot be resolved, `subprocess.Popen` raises an unhandled `FileNotFoundError` that crashes the Flask app. The UI shows no error feedback to the user.

**7.** The `strip_quotes()` helper removes all wrapping quotes from string flag values. Values that legitimately contain quotes would be corrupted. Currently low risk but unguarded.

**8.** The graph overhead estimate is hardcoded at `900.0 + (context_length × 0.001)` MB. Users with smaller GPUs (e.g. 6GB RTX 3050) receive over-conservative splits with no way to adjust the value.