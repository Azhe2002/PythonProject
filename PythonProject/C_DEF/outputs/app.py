from __future__ import annotations

import threading
from pathlib import Path
from tkinter import BOTH, DISABLED, LEFT, NORMAL, RIGHT, VERTICAL, filedialog, messagebox
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from ai_face_detector import AIFaceDetector, DetectionResult
from ai_face_detector.manual_roi import ManualFaceBox, normalize_box
from ai_face_detector.visualization import save_rgb


class DetectorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AI 生成人脸检测识别系统")
        self.geometry("1180x760")
        self.minsize(980, 640)

        self.detector = AIFaceDetector()
        self.current_path: Path | None = None
        self.current_result: DetectionResult | None = None
        self.preview_image: ImageTk.PhotoImage | None = None
        self.current_view = "annotated"

        self.original_image_size: tuple[int, int] | None = None
        self.display_source_size: tuple[int, int] | None = None
        self.display_image_origin = (0, 0)
        self.display_image_size = (0, 0)
        self.manual_selecting = False
        self.manual_box: ManualFaceBox | None = None
        self.drag_start_source: tuple[int, int] | None = None
        self.drag_rect_id: int | None = None

        self._build_style()
        self._build_layout()
        self._set_status("请选择一张人脸图像开始检测。")

    def _build_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#f5f7fb")
        style.configure("TLabel", background="#f5f7fb", foreground="#18202f", font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Score.TLabel", font=("Microsoft YaHei UI", 34, "bold"), foreground="#c0392b")
        style.configure("Subtle.TLabel", foreground="#5f6b7a")
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(10, 6))
        style.configure("Treeview", rowheight=27, font=("Microsoft YaHei UI", 9))
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"))

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill=BOTH, expand=True)

        left = ttk.Frame(root)
        left.pack(side=LEFT, fill=BOTH, expand=True)

        right = ttk.Frame(root, width=380)
        right.pack(side=RIGHT, fill="y", padx=(14, 0))
        right.pack_propagate(False)

        header = ttk.Frame(left)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="AI 生成人脸检测识别系统", style="Title.TLabel").pack(side=LEFT)
        mode = "训练模型" if self.detector.model_loaded else "物理启发式"
        ttk.Label(header, text=f"当前模式：{mode}", style="Subtle.TLabel").pack(side=RIGHT)

        self.image_canvas = tk.Canvas(left, bg="#ffffff", highlightthickness=1, highlightbackground="#d5dce8")
        self.image_canvas.pack(fill=BOTH, expand=True)
        self.image_canvas.create_text(400, 260, text="图像预览区", fill="#5f6b7a", font=("Microsoft YaHei UI", 14))
        self.image_canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.image_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        view_bar = ttk.Frame(left)
        view_bar.pack(fill="x", pady=(10, 0))
        ttk.Button(view_bar, text="标注图", command=lambda: self._show_result_image("annotated")).pack(side=LEFT)
        ttk.Button(view_bar, text="残差热力图", command=lambda: self._show_result_image("residual")).pack(side=LEFT, padx=6)
        ttk.Button(view_bar, text="频域响应图", command=lambda: self._show_result_image("frequency")).pack(side=LEFT)

        control = ttk.Frame(right)
        control.pack(fill="x")
        ttk.Button(control, text="选择图像", command=self._choose_image).pack(fill="x")
        self.detect_button = ttk.Button(control, text="开始检测", command=self._run_detection, state=DISABLED)
        self.detect_button.pack(fill="x", pady=8)
        ttk.Button(control, text="手动框选人脸", command=self._start_manual_selection).pack(fill="x")
        ttk.Button(control, text="清除手动框", command=self._clear_manual_box).pack(fill="x", pady=(8, 0))
        ttk.Button(control, text="导出报告", command=self._export_report).pack(fill="x", pady=(8, 0))

        ttk.Separator(right).pack(fill="x", pady=14)

        ttk.Label(right, text="AI 生成概率", style="Subtle.TLabel").pack(anchor="w")
        self.score_label = ttk.Label(right, text="--.-%", style="Score.TLabel")
        self.score_label.pack(anchor="w", pady=(0, 4))
        self.result_label = ttk.Label(right, text="等待检测", font=("Microsoft YaHei UI", 13, "bold"))
        self.result_label.pack(anchor="w")
        self.progress = ttk.Progressbar(right, maximum=100, length=340)
        self.progress.pack(fill="x", pady=(10, 12))

        self.meta_label = ttk.Label(right, text="", wraplength=350, style="Subtle.TLabel")
        self.meta_label.pack(anchor="w", fill="x")

        ttk.Label(right, text="分支证据").pack(anchor="w", pady=(16, 4))
        self.branch_tree = ttk.Treeview(right, columns=("branch", "real", "ai"), show="headings", height=4)
        self.branch_tree.heading("branch", text="分支")
        self.branch_tree.heading("real", text="真实证据")
        self.branch_tree.heading("ai", text="AI 证据")
        self.branch_tree.column("branch", width=150)
        self.branch_tree.column("real", width=85, anchor="center")
        self.branch_tree.column("ai", width=85, anchor="center")
        self.branch_tree.pack(fill="x")

        ttk.Label(right, text="解释说明").pack(anchor="w", pady=(16, 4))
        evidence_frame = ttk.Frame(right)
        evidence_frame.pack(fill=BOTH, expand=True)
        scrollbar = ttk.Scrollbar(evidence_frame, orient=VERTICAL)
        self.evidence_text = tk.Text(
            evidence_frame,
            height=10,
            wrap="word",
            font=("Microsoft YaHei UI", 9),
            bg="#ffffff",
            fg="#202936",
            relief="solid",
            borderwidth=1,
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=self.evidence_text.yview)
        self.evidence_text.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill="y")
        self.evidence_text.configure(state=DISABLED)

        self.status_label = ttk.Label(right, text="", wraplength=350, style="Subtle.TLabel")
        self.status_label.pack(fill="x", pady=(10, 0))

    def _choose_image(self) -> None:
        filetypes = [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
            ("所有文件", "*.*"),
        ]
        filename = filedialog.askopenfilename(title="选择待检测图像", filetypes=filetypes)
        if not filename:
            return
        self.current_path = Path(filename)
        self.detect_button.configure(state=NORMAL)
        self.current_result = None
        self.manual_box = None
        self.manual_selecting = False
        self.score_label.configure(text="--.-%")
        self.result_label.configure(text="等待检测")
        self.progress.configure(value=0)
        self.meta_label.configure(text="")
        self._show_file_preview(self.current_path)
        self._set_status(f"已选择：{self.current_path.name}")

    def _show_file_preview(self, path: Path) -> None:
        with Image.open(path) as img:
            image = img.convert("RGB")
            self.original_image_size = image.size
            self.current_view = "annotated"
            self._show_pil_image(image)

    def _show_pil_image(self, image: Image.Image) -> None:
        source_size = image.size
        self.display_source_size = source_size
        canvas_width = max(640, self.image_canvas.winfo_width())
        canvas_height = max(500, self.image_canvas.winfo_height())

        display = image.copy()
        display.thumbnail((canvas_width - 20, canvas_height - 20), Image.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(display)

        x0 = max(0, (canvas_width - display.width) // 2)
        y0 = max(0, (canvas_height - display.height) // 2)
        self.display_image_origin = (x0, y0)
        self.display_image_size = (display.width, display.height)

        self.image_canvas.delete("all")
        self.image_canvas.create_image(x0, y0, anchor="nw", image=self.preview_image)
        self._draw_manual_box_overlay()

    def _show_array(self, array) -> None:
        self._show_pil_image(Image.fromarray(array))

    def _start_manual_selection(self) -> None:
        if not self.current_path:
            messagebox.showinfo("提示", "请先选择一张图像。")
            return
        self.current_result = None
        self.manual_selecting = True
        self._show_file_preview(self.current_path)
        self._set_status("手动框选已开启：在图像上按住鼠标左键拖拽，松开后生成检测框。")

    def _clear_manual_box(self) -> None:
        self.manual_box = None
        self.manual_selecting = False
        self.drag_start_source = None
        self.drag_rect_id = None
        if self.current_result:
            self._show_result_image(self.current_view)
        elif self.current_path:
            self._show_file_preview(self.current_path)
        self._set_status("已清除手动框，后续检测将使用自动人脸定位。")

    def _canvas_to_source(self, x: int, y: int) -> tuple[int, int] | None:
        if not self.original_image_size or self.display_source_size != self.original_image_size:
            return None
        ox, oy = self.display_image_origin
        dw, dh = self.display_image_size
        if dw <= 0 or dh <= 0:
            return None
        if x < ox or x > ox + dw or y < oy or y > oy + dh:
            return None

        cx = min(max(x, ox), ox + dw)
        cy = min(max(y, oy), oy + dh)
        source_w, source_h = self.original_image_size
        sx = int(round((cx - ox) / dw * source_w))
        sy = int(round((cy - oy) / dh * source_h))
        return min(max(sx, 0), source_w), min(max(sy, 0), source_h)

    def _source_box_to_canvas(self, box: ManualFaceBox) -> tuple[int, int, int, int] | None:
        if not self.original_image_size or self.display_source_size != self.original_image_size:
            return None
        source_w, source_h = self.original_image_size
        ox, oy = self.display_image_origin
        dw, dh = self.display_image_size
        x0 = ox + int(round(box.x / source_w * dw))
        y0 = oy + int(round(box.y / source_h * dh))
        x1 = ox + int(round((box.x + box.w) / source_w * dw))
        y1 = oy + int(round((box.y + box.h) / source_h * dh))
        return x0, y0, x1, y1

    def _draw_manual_box_overlay(self) -> None:
        self.image_canvas.delete("manual_box")
        if not self.manual_box:
            return
        coords = self._source_box_to_canvas(self.manual_box)
        if not coords:
            return
        x0, y0, x1, y1 = coords
        self.image_canvas.create_rectangle(x0, y0, x1, y1, outline="#2c7be5", width=3, tags="manual_box")
        self.image_canvas.create_text(
            x0 + 6,
            max(12, y0 - 12),
            anchor="w",
            text="手动人脸框",
            fill="#2c7be5",
            font=("Microsoft YaHei UI", 10, "bold"),
            tags="manual_box",
        )

    def _on_canvas_press(self, event) -> None:
        if not self.manual_selecting:
            return
        point = self._canvas_to_source(event.x, event.y)
        if point is None:
            self._set_status("请在原图预览或标注图上进行手动框选。")
            return
        self.drag_start_source = point
        self.image_canvas.delete("drag_box")
        self.drag_rect_id = self.image_canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="#2c7be5",
            width=2,
            dash=(5, 3),
            tags="drag_box",
        )

    def _on_canvas_drag(self, event) -> None:
        if not self.manual_selecting or self.drag_rect_id is None:
            return
        ox, oy = self.display_image_origin
        dw, dh = self.display_image_size
        x = min(max(event.x, ox), ox + dw)
        y = min(max(event.y, oy), oy + dh)
        start = self.drag_start_source
        if start is None:
            return
        sx, sy = start
        source_w, source_h = self.original_image_size or (1, 1)
        x0 = ox + int(round(sx / source_w * dw))
        y0 = oy + int(round(sy / source_h * dh))
        self.image_canvas.coords(self.drag_rect_id, x0, y0, x, y)

    def _on_canvas_release(self, event) -> None:
        if not self.manual_selecting or self.drag_start_source is None:
            return
        end = self._canvas_to_source(event.x, event.y)
        start = self.drag_start_source
        self.drag_start_source = None
        self.image_canvas.delete("drag_box")
        self.drag_rect_id = None
        if end is None or self.original_image_size is None:
            return

        x0, y0 = start
        x1, y1 = end
        raw_box = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        source_w, source_h = self.original_image_size
        box = normalize_box(raw_box, (source_h, source_w, 3), min_size=24)
        if box is None:
            self._set_status("手动框太小，请重新拖拽一个覆盖完整人脸的矩形。")
            return

        self.manual_box = box
        self.manual_selecting = False
        self._draw_manual_box_overlay()
        self._set_status(f"已设置手动人脸框：x={box.x}, y={box.y}, w={box.w}, h={box.h}。点击“开始检测”即可使用。")

    def _run_detection(self) -> None:
        if not self.current_path:
            return
        self.detect_button.configure(state=DISABLED)
        box_text = "手动框选区域" if self.manual_box else "自动人脸定位区域"
        self._set_status(f"正在基于{box_text}提取频域残差和物理噪声特征...")
        manual_box = self.manual_box.as_tuple() if self.manual_box else None

        def worker() -> None:
            try:
                result = self.detector.analyze(self.current_path, manual_box=manual_box)
                self.after(0, lambda: self._set_result(result))
            except Exception as exc:
                self.after(0, lambda: self._show_error(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _set_result(self, result: DetectionResult) -> None:
        self.current_result = result
        self.detect_button.configure(state=NORMAL)
        self.score_label.configure(text=f"{result.ai_probability * 100:.1f}%")
        self.result_label.configure(text=f"{result.label}，置信度 {result.confidence * 100:.1f}%")
        self.progress.configure(value=result.ai_probability * 100)
        self.meta_label.configure(text=f"{result.face_note}\n模式：{result.model_mode}")

        for item in self.branch_tree.get_children():
            self.branch_tree.delete(item)
        for branch in result.branch_scores:
            self.branch_tree.insert(
                "",
                "end",
                values=(branch.name, f"{branch.real_evidence * 100:.1f}%", f"{branch.ai_evidence * 100:.1f}%"),
            )

        self.evidence_text.configure(state=NORMAL)
        self.evidence_text.delete("1.0", "end")
        self.evidence_text.insert("end", "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(result.evidence)))
        self.evidence_text.configure(state=DISABLED)
        self._show_result_image(self.current_view)
        self._set_status("检测完成。")

    def _show_result_image(self, view: str) -> None:
        self.current_view = view
        if not self.current_result:
            if self.current_path:
                self._show_file_preview(self.current_path)
            return
        if view == "residual":
            self._show_array(self.current_result.residual_image)
        elif view == "frequency":
            self._show_array(self.current_result.frequency_image)
        else:
            self._show_array(self.current_result.annotated_image)

    def _export_report(self) -> None:
        if not self.current_result or not self.current_path:
            messagebox.showinfo("提示", "请先完成一次检测。")
            return
        output_dir = Path("outputs") / self.current_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        save_rgb(output_dir / "annotated.jpg", self.current_result.annotated_image)
        save_rgb(output_dir / "residual_heatmap.jpg", self.current_result.residual_image)
        save_rgb(output_dir / "frequency_map.jpg", self.current_result.frequency_image)
        (output_dir / "report.json").write_text(self.current_result.to_json(), encoding="utf-8")
        self._set_status(f"报告已导出到：{output_dir.resolve()}")

    def _show_error(self, exc: Exception) -> None:
        self.detect_button.configure(state=NORMAL)
        self._set_status("检测失败。")
        messagebox.showerror("检测失败", str(exc))

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)


if __name__ == "__main__":
    DetectorApp().mainloop()
