# print.py打印limiter可用参数

from pedalboard import VST3Plugin

SHELL_PATH = r"WaveShell1-VST3 14.12_x64.vst3"
PLUGIN_NAME = "L1 limiter Stereo"

plugin = VST3Plugin(SHELL_PATH, plugin_name=PLUGIN_NAME)

print(f"Loaded: {PLUGIN_NAME}")
print(f"Param count: {len(plugin.parameters)}\n")


def fmt(v):
    try:
        return f"{v:.6g}" if isinstance(v, (int, float)) else str(v)
    except Exception:
        return repr(v)


fields_to_try = [
    "value",
    "raw_value",
    "default_value",
    "min_value",
    "max_value",
    "step_size",
    "units",
    "label",
]

for idx, (name, param) in enumerate(plugin.parameters.items(), start=1):
    print(f"{idx:03d}. {name}")
    for field in fields_to_try:
        if hasattr(param, field):
            print(f"    {field}: {fmt(getattr(param, field))}")
    if not any(hasattr(param, f) for f in fields_to_try):
        print(f"    (param): {param!r}")
    print()
