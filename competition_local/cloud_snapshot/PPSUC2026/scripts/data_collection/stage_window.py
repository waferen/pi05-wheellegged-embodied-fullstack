"""Independent stage display; never sends robot commands or changes recordings."""
import json
import re
import os
from pathlib import Path
import sys
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/ppsuc_rl_lab/ppsuc_rl_lab"))
from data_collection.stages import STAGE_PROMPTS_ZH, STAGE_COUNT



class StageWindow:
    def __init__(self, path, parent_pid):
        self.path, self.parent_pid = Path(path), parent_pid
        self.last = None
        self.root = tk.Tk()
        self.root.title("PPSUC 采集阶段")
        self.root.geometry("520x310+30+40")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#18212b")
        self.font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        self.panel = tk.Label(self.root, borderwidth=0, bg="#18212b")
        self.panel.pack(fill="both", expand=True)
        self.render(None)
        self.root.after(200, self.poll)
        # No focus_force/lift on updates: leave keyboard focus in the simulator.

    def render(self, data):
        # Conda Tk lacks Xft here. Render antialiased CJK text with Pillow instead.
        blocks = []
        if data is None:
            blocks.append(("等待仿真启动", 24, "white"))
        else:
            stage = int(data["stage"])
            message = data["message"]
            if data["error"]:
                status = "异常"
            elif message.startswith("Saved:"):
                status = "已保存"
            elif message.startswith("Recording ended:"):
                status = "未录制帧"
            elif data.get("recording"):
                status = "录制中"
            else:
                status = "等待录制"
            blocks.append((f"阶段 {stage} / {STAGE_COUNT} · {status}", 24, "#ff7676" if data["error"] else "#8cdbac"))
            mode_label = {"walk": "1 · 行走", "stair": "2 · 楼梯通行",
                          "low_grasp": "4 · 低位抓取", "high_grasp": "5 · 高位抓取"}.get(data.get("mode"), "等待同步")
            mode_color = "white"
            if data.get("mode") in ("low_grasp", "high_grasp"):
                ready = data.get("arm_ready", False)
                if ready:
                    mode_label += "｜机械臂可控"
                    mode_color = "#8cdbac"
                elif data.get("mode") == "low_grasp":
                    mode_label += "｜稳定中"
                    mode_color = "#ffd166"
            blocks.append((f"小脑模式：{mode_label}", 18, mode_color))
            blocks.append((STAGE_PROMPTS_ZH[stage-1], 19, "white"))
            # Stage log repeats both prompt and shortcuts; do not display it.
            if data["error"] or message.startswith("Saved:"):
                blocks.append((message, 14, "#ff7676" if data["error"] else "#8cdbac"))
        blocks.append(("F1 开始   F2 下一阶段   F3 保存", 15, "#9caaba"))
        image = Image.new("RGB", (560, 1600), "#18212b")
        draw = ImageDraw.Draw(image)
        y = 16
        for text, size, color in blocks:
            font = ImageFont.truetype(self.font_path, size, index=2)
            line = ""
            for char in re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]", text):
                if char == "\n" or draw.textlength(line+char, font=font) > 520:
                    draw.text((20,y),line,font=font,fill=color)
                    y += size+8
                    line = "" if char == "\n" else char.lstrip()
                else:
                    line += char
            if line:
                draw.text((20,y),line,font=font,fill=color)
                y += size+8
            y += 12
        image = image.crop((0,0,560,y+4))
        self.photo = ImageTk.PhotoImage(image)
        self.panel.configure(image=self.photo)
        self.root.geometry(f"560x{y+4}")

    def poll(self):
        try:
            os.kill(self.parent_pid, 0)
        except ProcessLookupError:
            self.root.destroy()
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data != self.last:
                self.render(data)
                self.last = data
        except (OSError, ValueError, KeyError, IndexError):
            pass
        self.root.after(200, self.poll)


if __name__ == "__main__":
    window = StageWindow(sys.argv[1], int(sys.argv[2]))
    window.root.mainloop()
