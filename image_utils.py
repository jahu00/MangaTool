"""Image analysis and splitting utilities.

The core idea: each input image is a double-page scan that must be cut in half.
Each half usually has some uniform-colored padding (scanner border) around the
real content. We detect that content, then crop every half to a *single common
box* so all output pages share consistent framing ("uniform padding").
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# Extensions we treat as images.
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp",
}

# Split ordering options.
LEFT_TO_RIGHT = "Left to Right"
RIGHT_TO_LEFT = "Right to Left"

# Crop area modes.
CROP_GLOBAL = "Global (uniform)"       # one consensus crop for every page
CROP_INDIVIDUAL = "Individual (per page)"  # each page cropped to its own content


@dataclass
class HalfResult:
    """Detection result for a single half of an image."""

    side: str  # "left" or "right"
    bbox: tuple[int, int, int, int] | None  # (left, top, right, bottom) or None
    size: tuple[int, int]  # (width, height) of the half

    @property
    def is_empty(self) -> bool:
        return self.bbox is None


@dataclass
class ImageAnalysis:
    """Detection result for one full image (both halves)."""

    path: str
    size: tuple[int, int]  # (width, height) of the source image
    left: HalfResult
    right: HalfResult

    def ordered_halves(self, order: str) -> list[HalfResult]:
        """Return halves in the requested output order."""
        if order == LEFT_TO_RIGHT:
            return [self.left, self.right]
        return [self.right, self.left]


def natural_key(name: str):
    """Sort helper so file2 comes before file10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_image_files(folder: str) -> list[str]:
    """Return sorted absolute paths of image files directly inside *folder*."""
    if not folder or not os.path.isdir(folder):
        return []
    names = [
        n for n in os.listdir(folder)
        if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS
        and os.path.isfile(os.path.join(folder, n))
    ]
    names.sort(key=natural_key)
    return [os.path.join(folder, n) for n in names]


def get_image_size(path: str) -> tuple[int, int]:
    """Return (width, height) without loading full pixel data."""
    with Image.open(path) as img:
        return img.size


def split_halves(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Split an image vertically into equal-width left and right halves.

    If the width is odd the single center column is dropped so both halves
    share identical dimensions, which keeps the uniform crop well defined.
    """
    w, h = img.size
    mid = w // 2
    left = img.crop((0, 0, mid, h))
    right = img.crop((w - mid, 0, w, h))
    return left, right


def _peel_edge(lines, threshold: int, skip: int = 0) -> int:
    """Count leading uniform padding lines in *lines*.

    *lines* is an array shaped (n_lines, n_pixels, 3) ordered from the outer
    edge inward. The first *skip* lines are treated as padding unconditionally
    -- this lets the caller ignore a band that contains UI/overlay artifacts
    which would otherwise stop detection early. The padding color is sampled
    just past the skipped band, then we walk further inward while each
    successive line stays within *threshold* of it. This keeps detection
    independent of the rest of the image.
    """
    n = len(lines)
    if n == 0:
        return 0
    skip = max(0, min(skip, n))
    if skip >= n:
        return n
    pad_color = np.median(lines[skip], axis=0)
    count = skip
    for line in lines[skip:]:
        diff = np.abs(line.astype(np.int16) - pad_color.astype(np.int16)).max()
        if diff > threshold:
            break
        count += 1
    return count


def content_bbox(
    arr: np.ndarray,
    threshold: int,
    skips: tuple[int, int, int, int] = (0, 0, 0, 0),
    detect_vertical: bool = False,
) -> tuple[int, int, int, int] | None:
    """Bounding box of real content, found by peeling padding from each edge.

    The left and right edges are always peeled independently using each edge's
    own padding color, so horizontal padding of any solid color works. The top
    and bottom edges are only peeled when *detect_vertical* is true; otherwise
    the full image height is kept.

    *skips* is ``(left, top, right, bottom)`` pixel bands to ignore (skip past)
    at each edge before detection, useful for UI elements sitting in the
    padding. Returns (left, top, right, bottom) or None if all padding.
    """
    h, w = arr.shape[0], arr.shape[1]
    skip_l, skip_t, skip_r, skip_b = skips

    left = _peel_edge(arr.transpose(1, 0, 2), threshold, skip_l)  # cols L->R
    right_pad = _peel_edge(
        arr[:, ::-1, :].transpose(1, 0, 2), threshold, skip_r
    )
    if detect_vertical:
        top = _peel_edge(arr, threshold, skip_t)  # rows T->B
        bottom_pad = _peel_edge(arr[::-1, :, :], threshold, skip_b)
    else:
        top = 0
        bottom_pad = 0

    right = w - right_pad
    bottom = h - bottom_pad
    if left >= right or top >= bottom:
        return None  # entirely padding
    return (left, top, right, bottom)


def _half_to_array(half: Image.Image) -> np.ndarray:
    if half.mode != "RGB":
        half = half.convert("RGB")
    return np.asarray(half)


def analyze_half(
    half: Image.Image,
    side: str,
    threshold: int,
    skip_outer: int = 0,
    skip_top: int = 0,
    skip_bottom: int = 0,
    detect_vertical: bool = False,
) -> HalfResult:
    """Detect content on a half, skipping a band at the outer/top/bottom edges.

    The *outer* edge is the left for a left half and the right for a right
    half; the inner (spine) edge is never skipped or cropped. Top/bottom
    margins are only detected when *detect_vertical* is true.
    """
    arr = _half_to_array(half)
    if side == "left":
        skips = (skip_outer, skip_top, 0, skip_bottom)  # (l, t, r, b)
    else:
        skips = (0, skip_top, skip_outer, skip_bottom)
    bbox = content_bbox(arr, threshold, skips, detect_vertical)
    return HalfResult(side=side, bbox=bbox, size=half.size)


def analyze_image(
    path: str,
    threshold: int,
    skip_outer: int = 0,
    skip_top: int = 0,
    skip_bottom: int = 0,
    detect_vertical: bool = False,
) -> ImageAnalysis:
    """Split one image and detect content bounds on each half.

    *skip_outer*/*skip_top*/*skip_bottom* ignore a pixel band at those edges
    before detection, so UI elements sitting in the margin don't cut it short.
    *detect_vertical* enables top/bottom margin detection (off by default, so
    only the left/right outer margins are trimmed).
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        left_img, right_img = split_halves(img)
        left = analyze_half(
            left_img, "left", threshold,
            skip_outer, skip_top, skip_bottom, detect_vertical,
        )
        right = analyze_half(
            right_img, "right", threshold,
            skip_outer, skip_top, skip_bottom, detect_vertical,
        )
        return ImageAnalysis(path=path, size=img.size, left=left, right=right)


def half_margins(half: HalfResult) -> tuple[int, int, int] | None:
    """Outer padding margins of a half as ``(outer, top, bottom)``.

    ``outer`` is the padding on the outward-facing side of the page: the left
    edge for a left half, the right edge for a right half. The inner edge (the
    spine / split line) is intentionally excluded -- only padding on the
    outside of the original image is measured. Returns None for empty halves.
    """
    if half.bbox is None:
        return None
    l, t, r, b = half.bbox
    w, h = half.size
    outer = l if half.side == "left" else (w - r)
    top = t
    bottom = h - b
    return (outer, top, bottom)


def _median_int(values: list[int]) -> int:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) // 2


def compute_uniform_margins(
    analyses: list[ImageAnalysis],
) -> tuple[int, int, int] | None:
    """Consensus outer margins ``(outer, top, bottom)`` across all halves.

    The padding is assumed uniform across the batch, so the median of each
    margin is used. The median naturally ignores one or two outliers (a half
    with unusually large or small padding) instead of letting them stretch the
    crop, which the previous union approach did.
    """
    outers: list[int] = []
    tops: list[int] = []
    bottoms: list[int] = []
    for a in analyses:
        for half in (a.left, a.right):
            m = half_margins(half)
            if m is not None:
                outers.append(m[0])
                tops.append(m[1])
                bottoms.append(m[2])
    if not outers:
        return None
    return (_median_int(outers), _median_int(tops), _median_int(bottoms))


def crop_box_from_margins(
    side: str, size: tuple[int, int], margins: tuple[int, int, int]
) -> tuple[int, int, int, int]:
    """Crop box for a half given consensus ``(outer, top, bottom)`` margins.

    Only the outer/top/bottom edges are trimmed; the inner (spine) edge is
    always kept so the split line is preserved.
    """
    outer, top, bottom = margins
    w, h = size
    top = max(0, min(top, h - 1))
    bottom = max(0, min(bottom, h - 1))
    if top + bottom >= h:  # degenerate: skip vertical crop
        top = bottom = 0
    outer = max(0, min(outer, w - 1))
    if side == "left":
        return (outer, top, w, h - bottom)  # keep inner (right) edge
    return (0, top, w - outer, h - bottom)  # keep inner (left) edge


def outer_crop_box_for_half(half: HalfResult) -> tuple[int, int, int, int] | None:
    """Per-half crop box using that half's own detected margins.

    This is the *individual* crop: only outer padding is trimmed and the inner
    (spine) edge is kept. Returns None for an empty half.
    """
    if half.bbox is None:
        return None
    l, t, r, b = half.bbox
    w, h = half.size
    if half.side == "left":
        return (l, t, w, b)  # keep inner (right) edge at the spine
    return (0, t, r, b)  # keep inner (left) edge at the spine


def crop_box_for_half(
    half: HalfResult,
    crop_mode: str,
    global_margins: tuple[int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    """Resolve the crop box for a half according to *crop_mode*.

    - ``CROP_GLOBAL``: apply the shared consensus *global_margins*.
    - ``CROP_INDIVIDUAL``: use the half's own detected content box.

    Returns None for an empty half (nothing to save).
    """
    if half.is_empty:
        return None
    if crop_mode == CROP_INDIVIDUAL:
        return outer_crop_box_for_half(half)
    if global_margins is None:
        return None
    return crop_box_from_margins(half.side, half.size, global_margins)


@dataclass
class ProcessResult:
    saved: int = 0
    skipped_empty: int = 0
    output_dir: str = ""      # folder written to (folder mode)
    cbz_path: str = ""        # archive written to (cbz mode)
    errors: list[str] = field(default_factory=list)


def default_cbz_name(input_dir: str) -> str:
    """CBZ filename derived from the input folder name."""
    base = os.path.basename(os.path.normpath(input_dir)) or "output"
    return base + ".cbz"


def normalize_cbz_name(name: str, input_dir: str) -> str:
    """Resolve a user-entered CBZ filename.

    Empty -> ``<input folder name>.cbz``. A name without the ``.cbz``
    extension gets it appended (case-insensitive check).
    """
    name = (name or "").strip()
    if not name:
        return default_cbz_name(input_dir)
    if not name.lower().endswith(".cbz"):
        name += ".cbz"
    return name


def process_folder(
    files: list[str],
    output_dir: str,
    order: str,
    threshold: int,
    crop_mode: str = CROP_GLOBAL,
    mode_overrides: dict[str, str] | None = None,
    create_cbz: bool = False,
    cbz_path: str | None = None,
    skip_outer: int = 0,
    skip_top: int = 0,
    skip_bottom: int = 0,
    detect_vertical: bool = False,
    progress=None,
) -> ProcessResult:
    """Analyze every file, then split and save according to the crop mode.

    *crop_mode* is the default applied to every page, either ``CROP_GLOBAL``
    (one consensus crop for the whole batch) or ``CROP_INDIVIDUAL`` (each page
    cropped to its own content). *mode_overrides* optionally maps individual
    file paths to a per-page mode that takes precedence over the default.

    The global consensus margins are always computed from the full batch, so a
    page set to ``CROP_GLOBAL`` uses the same crop whether or not other pages
    are overridden to individual.

    Output goes to a CBZ archive at *cbz_path* when *create_cbz* is true;
    otherwise pages are written as PNG files into *output_dir*.

    *progress* is an optional callback ``fn(done, total, message)``.
    """
    overrides = mode_overrides or {}

    def mode_for(path: str) -> str:
        return overrides.get(path, crop_mode)

    result = ProcessResult(output_dir=output_dir)

    # Set up the chosen output sink.
    zf = None
    if create_cbz:
        result.cbz_path = cbz_path or ""
        os.makedirs(os.path.dirname(os.path.abspath(cbz_path)), exist_ok=True)
        zf = zipfile.ZipFile(cbz_path, "w", zipfile.ZIP_DEFLATED)
    else:
        os.makedirs(output_dir, exist_ok=True)

    try:
        total = len(files)
        analyses: list[ImageAnalysis] = []
        for i, path in enumerate(files):
            if progress:
                progress(i, total, f"Analyzing {os.path.basename(path)}")
            try:
                analyses.append(
                    analyze_image(
                        path, threshold, skip_outer, skip_top, skip_bottom,
                        detect_vertical,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - report and continue
                result.errors.append(f"{os.path.basename(path)}: {exc}")

        # Consensus margins from the whole batch, used by any global-mode page.
        margins = compute_uniform_margins(analyses)

        # Count non-empty halves to determine zero-padding width.
        count = sum(
            1
            for a in analyses
            for half in (a.left, a.right)
            if not half.is_empty
        )
        pad_width = len(str(count)) if count > 0 else 1

        def emit(name: str, cropped: Image.Image):
            if zf is not None:
                buf = io.BytesIO()
                cropped.save(buf, format="PNG")
                zf.writestr(name, buf.getvalue())
            else:
                cropped.save(os.path.join(output_dir, name))

        index = 0
        for i, a in enumerate(analyses):
            if progress:
                progress(i, len(analyses), f"Saving {os.path.basename(a.path)}")
            try:
                with Image.open(a.path) as img:
                    img = img.convert("RGB")
                    left_img, right_img = split_halves(img)
                    halves = {"left": left_img, "right": right_img}
                    page_mode = mode_for(a.path)
                    for half_res in a.ordered_halves(order):
                        box = crop_box_for_half(half_res, page_mode, margins)
                        if box is None:
                            result.skipped_empty += 1
                            continue
                        src = halves[half_res.side]
                        cropped = src.crop(box)
                        index += 1
                        name = f"{str(index).zfill(pad_width)}.png"
                        emit(name, cropped)
                        result.saved += 1
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"{os.path.basename(a.path)}: {exc}")
    finally:
        if zf is not None:
            zf.close()

    if progress:
        progress(total, total, "Done")
    return result
