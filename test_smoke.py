"""Smoke test for image_utils: generate synthetic double-page scans,
run the full pipeline, and check the outputs."""

import os
import tempfile

from PIL import Image

import image_utils as iu


def make_scan(path, w, h, bg=(255, 255, 255), left_content=True, right_content=True):
    img = Image.new("RGB", (w, h), bg)
    mid = w // 2
    # draw content rectangles well inside the padding on each half
    px = img.load()
    def fill(x0, y0, x1, y1, color):
        for x in range(x0, x1):
            for y in range(y0, y1):
                px[x, y] = color
    if left_content:
        fill(30, 40, mid - 30, h - 40, (10, 10, 10))
    if right_content:
        fill(mid + 30, 40, w - 30, h - 40, (20, 20, 20))
    img.save(path)


def main():
    with tempfile.TemporaryDirectory() as d:
        make_scan(os.path.join(d, "a.png"), 400, 600)
        make_scan(os.path.join(d, "b.png"), 400, 600)
        # one image whose right half is pure padding
        make_scan(os.path.join(d, "c.png"), 400, 600, right_content=False)

        files = iu.list_image_files(d)
        assert len(files) == 3, files

        out = os.path.join(d, "output")
        res = iu.process_folder(
            files, out, iu.RIGHT_TO_LEFT, threshold=25,
            crop_mode=iu.CROP_GLOBAL,
        )
        print("saved:", res.saved, "skipped_empty:", res.skipped_empty)
        print("errors:", res.errors)

        # 3 files * 2 halves = 6, minus 1 empty half = 5
        assert res.saved == 5, res.saved
        assert res.skipped_empty == 1, res.skipped_empty

        names = sorted(os.listdir(out))
        print("output files:", names)
        # zero-padded, single digit count -> width 1
        assert names == ["1.png", "2.png", "3.png", "4.png", "5.png"], names

        # all outputs same size (consensus outer-margin crop)
        sizes = {Image.open(os.path.join(out, n)).size for n in names}
        assert len(sizes) == 1, sizes
        print("uniform output size:", sizes)

        # Verify only outer padding was removed: content is 30px inset on the
        # outer/top/bottom edges, so a half (width 200, height 600) crops to
        # width 200-30=170 and height 600-40-40=520.
        assert sizes == {(170, 520)}, sizes

        # Outlier test: one half with much larger padding must not shrink the
        # others (median ignores it).
        with tempfile.TemporaryDirectory() as d3:
            make_scan(os.path.join(d3, "n1.png"), 400, 600)
            make_scan(os.path.join(d3, "n2.png"), 400, 600)
            make_scan(os.path.join(d3, "n3.png"), 400, 600)
            # outlier: content pushed far inward on both halves
            img = Image.new("RGB", (400, 600), (255, 255, 255))
            px = img.load()
            for x in range(90, 110):
                for y in range(200, 400):
                    px[x, y] = (0, 0, 0)
            for x in range(290, 310):
                for y in range(200, 400):
                    px[x, y] = (0, 0, 0)
            img.save(os.path.join(d3, "n4_outlier.png"))
            all_files = iu.list_image_files(d3)
            out3 = os.path.join(d3, "output")
            iu.process_folder(
                all_files, out3, iu.LEFT_TO_RIGHT, threshold=25,
                crop_mode=iu.CROP_GLOBAL,
            )
            sizes3 = {Image.open(os.path.join(out3, n)).size
                      for n in os.listdir(out3)}
            # Consensus should follow the 3 normal images (170x520), unaffected
            # by the single outlier.
            assert sizes3 == {(170, 520)}, sizes3
            print("outlier-resistant (global) size:", sizes3)

            # Individual mode: each page cropped to its own content, so the
            # outlier keeps its own (smaller) size while normals are 170x520.
            out3i = os.path.join(d3, "output_individual")
            iu.process_folder(
                all_files, out3i, iu.LEFT_TO_RIGHT, threshold=25,
                crop_mode=iu.CROP_INDIVIDUAL,
            )
            sizes3i = {Image.open(os.path.join(out3i, n)).size
                       for n in os.listdir(out3i)}
            # Normal halves -> 170x520; outlier halves crop tightly to their
            # own 20x200 content, so we should see more than one size.
            assert (170, 520) in sizes3i, sizes3i
            assert len(sizes3i) > 1, sizes3i
            print("individual sizes:", sorted(sizes3i))

            # Per-page override: default global, but force the outlier to
            # individual. Normal pages stay 170x520; only the outlier's two
            # halves crop tightly to 110x200.
            outlier = [f for f in all_files if "outlier" in f][0]
            out3o = os.path.join(d3, "output_override")
            iu.process_folder(
                all_files, out3o, iu.LEFT_TO_RIGHT, threshold=25,
                crop_mode=iu.CROP_GLOBAL,
                mode_overrides={outlier: iu.CROP_INDIVIDUAL},
            )
            all_sizes = [Image.open(os.path.join(out3o, n)).size
                         for n in os.listdir(out3o)]
            assert all_sizes.count((170, 520)) == 6, all_sizes  # 3 normals x2
            assert all_sizes.count((110, 200)) == 2, all_sizes  # outlier x2
            print("override sizes:", sorted(set(all_sizes)))

        # test zero padding with many outputs
        for i in range(6):
            make_scan(os.path.join(d, f"z{i}.png"), 400, 600)
        files2 = iu.list_image_files(d)
        out2 = os.path.join(d, "output2")
        res2 = iu.process_folder(
            files2, out2, iu.LEFT_TO_RIGHT, threshold=25,
            crop_mode=iu.CROP_GLOBAL,
        )
        names2 = sorted(os.listdir(out2))
        # count > 9 -> width 2 -> names like 01.png
        assert names2[0] == "01.png", names2[0]
        print("padded first name:", names2[0], "total:", res2.saved)

        # CBZ name normalization
        assert iu.normalize_cbz_name("", "/a/b/MyManga") == "MyManga.cbz"
        assert iu.normalize_cbz_name("vol1", "/a/b/x") == "vol1.cbz"
        assert iu.normalize_cbz_name("vol1.cbz", "/a/b/x") == "vol1.cbz"
        assert iu.normalize_cbz_name("vol1.CBZ", "/a/b/x") == "vol1.CBZ"
        print("cbz name normalization OK")

        # CBZ output produces a valid zip with the expected zero-padded pages.
        import zipfile
        cbz_dir = os.path.join(d, "cbz_test")
        os.makedirs(cbz_dir)
        for i in range(3):
            make_scan(os.path.join(cbz_dir, f"{i+1}.png"), 400, 600)
        cbz_path = os.path.join(cbz_dir, "book.cbz")
        rescbz = iu.process_folder(
            iu.list_image_files(cbz_dir), "", iu.LEFT_TO_RIGHT, threshold=25,
            crop_mode=iu.CROP_GLOBAL, create_cbz=True, cbz_path=cbz_path,
        )
        assert os.path.isfile(cbz_path), cbz_path
        assert rescbz.cbz_path == cbz_path
        with zipfile.ZipFile(cbz_path) as zf:
            entries = sorted(zf.namelist())
        # 3 files x 2 halves = 6 pages -> width 1
        assert entries == ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png"], entries
        print("cbz entries:", entries)

        # Skip-margin: a UI element sits in the outer padding and would stop
        # detection early; skipping past it recovers the true margin.
        skip_dir = os.path.join(d, "skip_test")
        os.makedirs(skip_dir)
        # 400x600 image, gray(51) padding, small UI blob near the left edge,
        # real content well inside.
        img = Image.new("RGB", (400, 600), (51, 51, 51))
        px = img.load()
        for x in range(10, 25):          # UI element in the left padding
            for y in range(280, 320):
                px[x, y] = (200, 200, 200)
        for x in range(120, 190):        # real content (left half is 0..200)
            for y in range(60, 540):
                px[x, y] = (255, 255, 255)
        for x in range(210, 280):        # real content in right half
            for y in range(60, 540):
                px[x, y] = (255, 255, 255)
        img.save(os.path.join(skip_dir, "1.png"))

        a0 = iu.analyze_image(os.path.join(skip_dir, "1.png"), 25)
        # Without skip, the UI blob at x=10 stops peeling early -> tiny margin.
        assert iu.half_margins(a0.left)[0] < 30, iu.half_margins(a0.left)
        a1 = iu.analyze_image(os.path.join(skip_dir, "1.png"), 25, 30, 0, 0)
        # Skipping 30px past the blob recovers the real 120px left margin.
        assert iu.half_margins(a1.left)[0] == 120, iu.half_margins(a1.left)
        print("skip margin: without", iu.half_margins(a0.left)[0],
              "with", iu.half_margins(a1.left)[0])

    print("ALL OK")


if __name__ == "__main__":
    main()
