"""Tkinter UI for splitting double-page image scans into single pages.

Workflow:
1. Pick an input folder. The file list shows each image and its height.
   Any image whose height differs from the "desired height" is flagged red.
2. Optionally pick an output folder (defaults to the input folder). An
   "output" subfolder is created inside it for the results.
3. Choose the split order and preview how padding was detected / where the
   cut happens.
4. Process: every image is split in half, cropped to a uniform box, and saved
   as zero-padded PNGs (1.png -> 001.png, etc.).
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

import image_utils as iu


class SplitterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Splitter")
        self.geometry("1000x640")
        self.minsize(880, 560)

        # State
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.desired_height = tk.StringVar(value="0")
        self.order = tk.StringVar(value=iu.RIGHT_TO_LEFT)
        self.crop_mode = tk.StringVar(value=iu.CROP_GLOBAL)
        self.threshold = tk.IntVar(value=25)
        self.create_cbz = tk.BooleanVar(value=True)
        self.cbz_name = tk.StringVar()
        # Pixel bands to skip before margin detection (to ignore UI elements).
        self.skip_outer = tk.IntVar(value=0)
        self.skip_top = tk.IntVar(value=0)
        self.skip_bottom = tk.IntVar(value=0)
        # Optional top/bottom margin detection (off by default).
        self.detect_vertical = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Select an input folder to begin.")

        self.files: list[str] = []
        self.sizes: dict[str, tuple[int, int]] = {}
        # Per-page crop overrides: path -> CROP_GLOBAL/CROP_INDIVIDUAL.
        # Absent means "follow the batch default".
        self.mode_overrides: dict[str, str] = {}
        self._preview_photo = None  # keep a reference so it isn't GC'd

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=10)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        # Input folder
        ttk.Label(top, text="Input folder:").grid(row=0, column=0, sticky="w")
        input_entry = ttk.Entry(top, textvariable=self.input_dir)
        input_entry.grid(row=0, column=1, sticky="ew", padx=6)
        input_entry.bind("<Return>", lambda _e: self.load_files())
        ttk.Button(top, text="Browse...", command=self.choose_input).grid(
            row=0, column=2
        )
        ttk.Button(top, text="Load", command=self.load_files).grid(
            row=0, column=3, padx=(6, 0)
        )

        # Output folder
        self.output_label = ttk.Label(top, text="Output folder:")
        self.output_label.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.output_entry = ttk.Entry(top, textvariable=self.output_dir)
        self.output_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        self.output_btn = ttk.Button(
            top, text="Browse...", command=self.choose_output
        )
        self.output_btn.grid(row=1, column=2, pady=(6, 0))
        self.output_hint = ttk.Label(
            top, text="(empty = same as input)", foreground="#888"
        )
        self.output_hint.grid(
            row=1, column=3, sticky="w", padx=(6, 0), pady=(6, 0)
        )

        # CBZ output
        cbz_row = ttk.Frame(top)
        cbz_row.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Checkbutton(
            cbz_row,
            text="Create CBZ",
            variable=self.create_cbz,
            command=self._on_cbz_toggle,
        ).pack(side="left")
        ttk.Label(cbz_row, text="CBZ file name:").pack(side="left", padx=(12, 0))
        self.cbz_entry = ttk.Entry(cbz_row, textvariable=self.cbz_name, width=30)
        self.cbz_entry.pack(side="left", padx=(4, 6))
        ttk.Label(
            cbz_row, text="(empty = input folder name)", foreground="#888"
        ).pack(side="left")

        # Options row
        opts = ttk.Frame(top)
        opts.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(10, 0))

        ttk.Label(opts, text="Desired height:").pack(side="left")
        h_entry = ttk.Entry(opts, textvariable=self.desired_height, width=8)
        h_entry.pack(side="left", padx=(4, 16))
        h_entry.bind("<KeyRelease>", lambda _e: self.refresh_height_flags())

        ttk.Label(opts, text="Split order:").pack(side="left")
        ttk.Combobox(
            opts,
            textvariable=self.order,
            values=[iu.RIGHT_TO_LEFT, iu.LEFT_TO_RIGHT],
            state="readonly",
            width=14,
        ).pack(side="left", padx=(4, 16))

        ttk.Label(opts, text="Default crop area:").pack(side="left")
        crop_cb = ttk.Combobox(
            opts,
            textvariable=self.crop_mode,
            values=[iu.CROP_GLOBAL, iu.CROP_INDIVIDUAL],
            state="readonly",
            width=18,
        )
        crop_cb.pack(side="left", padx=(4, 16))
        crop_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_default_mode_change())

        ttk.Label(opts, text="Padding sensitivity:").pack(side="left")
        ttk.Scale(
            opts,
            from_=0,
            to=100,
            variable=self.threshold,
            orient="horizontal",
            length=140,
            command=self._on_threshold_change,
        ).pack(side="left", padx=(4, 4))
        ttk.Label(opts, textvariable=self.threshold, width=3).pack(side="left")

        # Skip-margin row: ignore a pixel band at each edge before detection,
        # so UI elements sitting in the padding don't cut margin detection
        # short. The inner (spine) edge is never affected.
        skip_row = ttk.Frame(top)
        skip_row.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(
            skip_row,
            text="Detect top/bottom margins",
            variable=self.detect_vertical,
            command=self._on_detect_vertical_toggle,
        ).pack(side="left", padx=(0, 16))
        ttk.Label(
            skip_row, text="Skip margin (px) before detection \u2014"
        ).pack(side="left")
        self._skip_widgets = {}
        for key, label, var in (
            ("outer", "Outer:", self.skip_outer),
            ("top", "Top:", self.skip_top),
            ("bottom", "Bottom:", self.skip_bottom),
        ):
            lbl = ttk.Label(skip_row, text=label)
            lbl.pack(side="left", padx=(12, 2))
            sb = ttk.Spinbox(
                skip_row,
                from_=0,
                to=100000,
                increment=5,
                textvariable=var,
                width=7,
                command=self._on_skip_change,
            )
            sb.pack(side="left")
            self._skip_widgets[key] = (lbl, sb)
        ttk.Label(
            skip_row,
            text="(use for UI/overlays in the margin)",
            foreground="#888",
        ).pack(side="left", padx=(12, 0))

        # Main split area: file list (left) + preview (right)
        main = ttk.Panedwindow(self, orient="horizontal")
        main.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 6))

        # File list
        left_frame = ttk.Frame(main)
        left_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)

        columns = ("name", "height", "crop")
        self.tree = ttk.Treeview(
            left_frame, columns=columns, show="headings", selectmode="browse"
        )
        self.tree.heading("name", text="File")
        self.tree.heading("height", text="Height")
        self.tree.heading("crop", text="Crop")
        self.tree.column("name", width=210, anchor="w")
        self.tree.column("height", width=60, anchor="center")
        self.tree.column("crop", width=90, anchor="center")
        self.tree.tag_configure("mismatch", background="#f7b7b7")
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self.update_preview())
        # Click the Crop column to cycle a page's override; right-click for menu.
        self.tree.bind("<Button-1>", self._on_tree_click)
        self.tree.bind("<Button-3>", self._on_tree_right_click)

        vsb = ttk.Scrollbar(
            left_frame, orient="vertical", command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.grid(row=0, column=1, sticky="ns")
        ttk.Label(
            left_frame,
            text="Click the Crop cell to cycle a page's mode (right-click for options).",
            foreground="#888",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        main.add(left_frame, weight=1)

        # Right-click menu for setting per-page crop mode.
        self.crop_menu = tk.Menu(self, tearoff=0)
        self.crop_menu.add_command(
            label="Use default",
            command=lambda: self._set_override(None),
        )
        self.crop_menu.add_command(
            label=f"Force {iu.CROP_GLOBAL}",
            command=lambda: self._set_override(iu.CROP_GLOBAL),
        )
        self.crop_menu.add_command(
            label=f"Force {iu.CROP_INDIVIDUAL}",
            command=lambda: self._set_override(iu.CROP_INDIVIDUAL),
        )
        self._menu_target: str | None = None

        # Preview
        right_frame = ttk.Frame(main)
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)
        ttk.Label(
            right_frame,
            text="Preview (green = crop area, blue = split line)",
        ).grid(row=0, column=0, sticky="w")
        self.preview = tk.Label(
            right_frame, background="#2b2b2b", anchor="center"
        )
        self.preview.grid(row=1, column=0, sticky="nsew")
        self.preview.bind("<Configure>", lambda _e: self._render_preview())
        main.add(right_frame, weight=2)

        # Bottom bar
        bottom = ttk.Frame(self, padding=(10, 4, 10, 10))
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        ttk.Label(bottom, textvariable=self.status).grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )
        self.process_btn = ttk.Button(
            bottom, text="Process", command=self.process
        )
        self.process_btn.grid(row=0, column=1)

        # cache of the last analyzed preview to avoid recomputation on resize
        self._preview_base = None  # PIL.Image already annotated

        # Apply initial enabled/disabled state for the output widgets.
        self._on_cbz_toggle()
        self._on_detect_vertical_toggle()

    # ------------------------------------------------------------ actions
    def _on_cbz_toggle(self):
        """Enable the CBZ name field or the output-folder field per checkbox."""
        cbz = self.create_cbz.get()
        folder_state = "disabled" if cbz else "normal"
        cbz_state = "normal" if cbz else "disabled"
        for w in (self.output_entry, self.output_btn):
            w.configure(state=folder_state)
        fg = "#bbb" if cbz else "#000"
        self.output_label.configure(foreground=fg)
        self.output_hint.configure(foreground="#bbb" if cbz else "#888")
        self.cbz_entry.configure(state=cbz_state)

    def choose_input(self):
        folder = filedialog.askdirectory(title="Select input folder")
        if folder:
            self.input_dir.set(folder)
            self.load_files()

    def choose_output(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_dir.set(folder)

    def load_files(self):
        folder = self.input_dir.get().strip()
        self.files = iu.list_image_files(folder)
        self.sizes.clear()
        self.mode_overrides.clear()
        self.tree.delete(*self.tree.get_children())

        if not folder or not os.path.isdir(folder):
            self.status.set(f"Folder not found: {folder}" if folder
                            else "Enter an input folder path.")
            return
        if not self.files:
            self.status.set("No image files found in that folder.")
            return

        for path in self.files:
            try:
                size = iu.get_image_size(path)
            except Exception:  # noqa: BLE001 - unreadable file
                size = (0, 0)
            self.sizes[path] = size
            self.tree.insert(
                "", "end", iid=path,
                values=(os.path.basename(path), size[1], self._crop_label(path)),
            )

        # Default desired height to the most common height.
        heights = [s[1] for s in self.sizes.values() if s[1] > 0]
        if heights:
            common = max(set(heights), key=heights.count)
            self.desired_height.set(str(common))

        self.refresh_height_flags()
        self.status.set(f"Loaded {len(self.files)} file(s).")
        # Select first row to show a preview.
        first = self.tree.get_children()
        if first:
            self.tree.selection_set(first[0])

    def _desired_height_value(self) -> int:
        try:
            return int(self.desired_height.get())
        except (ValueError, TypeError):
            return 0

    def refresh_height_flags(self):
        desired = self._desired_height_value()
        for path in self.files:
            h = self.sizes.get(path, (0, 0))[1]
            tags = ()
            if desired > 0 and h != desired:
                tags = ("mismatch",)
            if self.tree.exists(path):
                self.tree.item(path, tags=tags)

    # -------------------------------------------------- per-page crop mode
    @staticmethod
    def _short_mode(mode: str) -> str:
        return "Global" if mode == iu.CROP_GLOBAL else "Individual"

    def _crop_label(self, path: str) -> str:
        """Crop column text: the override, or the default shown as 'Default'."""
        override = self.mode_overrides.get(path)
        if override is None:
            return f"Default ({self._short_mode(self.crop_mode.get())})"
        return self._short_mode(override)

    def _effective_mode(self, path: str) -> str:
        return self.mode_overrides.get(path, self.crop_mode.get())

    def _refresh_crop_cell(self, path: str):
        if self.tree.exists(path):
            self.tree.set(path, "crop", self._crop_label(path))

    def _refresh_all_crop_cells(self):
        for path in self.files:
            self._refresh_crop_cell(path)

    def _on_default_mode_change(self):
        # Pages without an override display/use the new default.
        self._refresh_all_crop_cells()
        self.update_preview()

    def _set_override(self, mode: str | None):
        """Apply *mode* (or None = follow default) to the current menu target."""
        path = self._menu_target
        if path is None:
            return
        if mode is None:
            self.mode_overrides.pop(path, None)
        else:
            self.mode_overrides[path] = mode
        self._refresh_crop_cell(path)
        # If the changed page is selected, refresh its preview.
        sel = self.tree.selection()
        if sel and sel[0] == path:
            self.update_preview()

    def _cycle_override(self, path: str):
        """Cycle: default -> Global -> Individual -> default."""
        current = self.mode_overrides.get(path)
        if current is None:
            self.mode_overrides[path] = iu.CROP_GLOBAL
        elif current == iu.CROP_GLOBAL:
            self.mode_overrides[path] = iu.CROP_INDIVIDUAL
        else:
            self.mode_overrides.pop(path, None)
        self._refresh_crop_cell(path)
        sel = self.tree.selection()
        if sel and sel[0] == path:
            self.update_preview()

    def _on_tree_click(self, event):
        # Only react to clicks in the Crop column; let normal selection happen
        # otherwise.
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        if self.tree.identify_column(event.x) != "#3":  # crop is 3rd column
            return
        row = self.tree.identify_row(event.y)
        if row:
            self._cycle_override(row)
            return "break"  # don't change selection on a crop-cell click

    def _on_tree_right_click(self, event):
        row = self.tree.identify_row(event.y)
        if not row:
            return
        self._menu_target = row
        self.crop_menu.tk_popup(event.x_root, event.y_root)

    # ------------------------------------------------------------ preview
    def _on_threshold_change(self, value):
        self.threshold.set(int(float(value)))
        # Debounce: only refresh the preview once the slider settles.
        if getattr(self, "_thr_after", None) is not None:
            self.after_cancel(self._thr_after)
        self._thr_after = self.after(250, self._threshold_settled)

    def _threshold_settled(self):
        self._thr_after = None
        self._gm_key = None  # invalidate cached global margins
        self.update_preview()

    def _skip_values(self) -> tuple[int, int, int]:
        def val(var):
            try:
                return max(0, int(var.get()))
            except (ValueError, TypeError):
                return 0
        return val(self.skip_outer), val(self.skip_top), val(self.skip_bottom)

    def _on_skip_change(self):
        self._gm_key = None  # skips changed -> recompute consensus margins
        self.update_preview()

    def _on_detect_vertical_toggle(self):
        # Top/bottom skip only matter when vertical detection is enabled.
        state = "normal" if self.detect_vertical.get() else "disabled"
        fg = "#000" if self.detect_vertical.get() else "#bbb"
        for key in ("top", "bottom"):
            lbl, sb = self._skip_widgets[key]
            sb.configure(state=state)
            lbl.configure(foreground=fg)
        self._gm_key = None
        self.update_preview()

    def update_preview(self):
        sel = self.tree.selection()
        if not sel:
            return
        path = sel[0]
        threshold = self.threshold.get()
        crop_mode = self._effective_mode(path)
        skips = self._skip_values()
        detect_v = self.detect_vertical.get()

        def work():
            try:
                base = self._build_preview_image(
                    path, threshold, crop_mode, skips, detect_v
                )
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self.status.set(f"Preview error: {exc}"))
                return
            self._preview_base = base
            self.after(0, self._render_preview)

        threading.Thread(target=work, daemon=True).start()

    def _global_margins(
        self, threshold: int, skips: tuple[int, int, int], detect_v: bool
    ):
        """Consensus margins across all files, cached per input parameters."""
        key = (tuple(self.files), threshold, skips, detect_v)
        if getattr(self, "_gm_key", None) == key:
            return self._gm_value
        analyses = []
        for p in self.files:
            try:
                analyses.append(
                    iu.analyze_image(p, threshold, *skips, detect_v)
                )
            except Exception:  # noqa: BLE001 - skip unreadable in preview
                continue
        value = iu.compute_uniform_margins(analyses)
        self._gm_key = key
        self._gm_value = value
        return value

    def _build_preview_image(
        self, path: str, threshold: int, crop_mode: str,
        skips: tuple[int, int, int], detect_v: bool,
    ) -> Image.Image:
        """Return a copy of the source annotated with detection overlays.

        The green crop rectangle reflects the *selected crop mode*: the page's
        own content box in individual mode, or the shared consensus box in
        global mode (so the preview matches what will actually be saved).
        """
        analysis = iu.analyze_image(path, threshold, *skips, detect_v)
        global_margins = (
            self._global_margins(threshold, skips, detect_v)
            if crop_mode == iu.CROP_GLOBAL
            else None
        )
        with Image.open(path) as img:
            img = img.convert("RGB")
            annotated = img.convert("RGBA")
        draw = ImageDraw.Draw(annotated, "RGBA")
        w, h = annotated.size
        mid = w // 2
        skip_outer, skip_top, skip_bottom = skips

        # Skipped bands (semi-transparent orange), shown on the outer/top/
        # bottom edges of each half so the ignored region is visible.
        shade = (255, 140, 0, 90)
        if skip_outer > 0:
            draw.rectangle([0, 0, skip_outer, h], fill=shade)          # L outer
            draw.rectangle([w - skip_outer, 0, w, h], fill=shade)      # R outer
        if detect_v and skip_top > 0:
            draw.rectangle([0, 0, w, skip_top], fill=shade)
        if detect_v and skip_bottom > 0:
            draw.rectangle([0, h - skip_bottom, w, h], fill=shade)

        # Split line
        draw.line([(mid, 0), (mid, h)], fill=(70, 130, 255), width=max(2, w // 400))

        # Crop boxes (outer padding trimmed, spine edge kept), offset to their
        # position in the full image. Box depends on the selected crop mode.
        offsets = {"left": 0, "right": w - mid}
        for half in (analysis.left, analysis.right):
            box = iu.crop_box_for_half(half, crop_mode, global_margins)
            if box is None:
                continue
            ox = offsets[half.side]
            l, t, r, b = box
            draw.rectangle(
                [l + ox, t, r + ox - 1, b - 1],
                outline=(60, 220, 60),
                width=max(2, w // 400),
            )
        return annotated

    def _render_preview(self):
        if self._preview_base is None:
            return
        avail_w = max(self.preview.winfo_width(), 50)
        avail_h = max(self.preview.winfo_height(), 50)
        img = self._preview_base.copy()
        img.thumbnail((avail_w, avail_h), Image.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(img)
        self.preview.configure(image=self._preview_photo)

    # ------------------------------------------------------------ process
    def _resolve_output_dir(self) -> str:
        base = self.output_dir.get().strip() or self.input_dir.get().strip()
        return os.path.join(base, "output")

    def _resolve_cbz_path(self) -> str:
        """Full CBZ path, placed inside the input folder."""
        input_dir = self.input_dir.get().strip()
        name = iu.normalize_cbz_name(self.cbz_name.get(), input_dir)
        return os.path.join(input_dir, name)

    def process(self):
        if not self.files:
            messagebox.showwarning("No files", "Load an input folder first.")
            return
        order = self.order.get()
        threshold = self.threshold.get()
        crop_mode = self.crop_mode.get()
        overrides = dict(self.mode_overrides)
        create_cbz = self.create_cbz.get()
        out_dir = self._resolve_output_dir()
        cbz_path = self._resolve_cbz_path() if create_cbz else None
        skip_outer, skip_top, skip_bottom = self._skip_values()
        detect_vertical = self.detect_vertical.get()

        self.process_btn.configure(state="disabled")
        self.progress.configure(value=0, maximum=len(self.files) * 2)

        def on_progress(done, total, msg):
            self.after(0, lambda: self._set_progress(done, total, msg))

        def work():
            try:
                result = iu.process_folder(
                    self.files, out_dir, order, threshold,
                    crop_mode=crop_mode, mode_overrides=overrides,
                    create_cbz=create_cbz, cbz_path=cbz_path,
                    skip_outer=skip_outer, skip_top=skip_top,
                    skip_bottom=skip_bottom, detect_vertical=detect_vertical,
                    progress=on_progress,
                )
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._process_failed(exc))
                return
            self.after(0, lambda: self._process_done(result))

        threading.Thread(target=work, daemon=True).start()

    def _set_progress(self, done, total, msg):
        # Analysis then saving are two passes; scale bar over both.
        self.progress.configure(maximum=max(total, 1))
        self.progress.configure(value=done)
        self.status.set(msg)

    def _process_failed(self, exc):
        self.process_btn.configure(state="normal")
        messagebox.showerror("Processing failed", str(exc))
        self.status.set("Processing failed.")

    def _process_done(self, result: iu.ProcessResult):
        self.process_btn.configure(state="normal")
        self.progress.configure(value=self.progress["maximum"])
        destination = result.cbz_path or result.output_dir
        summary = (
            f"Saved {result.saved} page(s) to {destination}. "
            f"Skipped {result.skipped_empty} empty half/halves."
        )
        self.status.set(summary)
        if result.errors:
            messagebox.showwarning(
                "Completed with errors",
                summary + "\n\nErrors:\n" + "\n".join(result.errors[:20]),
            )
        else:
            messagebox.showinfo("Done", summary)


if __name__ == "__main__":
    SplitterApp().mainloop()
