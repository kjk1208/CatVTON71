# make_vitonhd_csv.py
import os, csv, argparse

# 마스크 후보 위치와 확장자(없으면 순서대로 탐색)
CANDIDATE_MASK_DIRS = ["agnostic-mask", "agnostic-v3.2"]
CANDIDATE_EXTS = [".png", ".jpg", ".jpeg"]

def find_mask(root_split_dir, person_rel, preferred_dir=None, preferred_ext=".png"):
    stem = os.path.splitext(person_rel)[0]  # image/000001_0.jpg -> image/000001_0
    dirs = []
    if preferred_dir: dirs.append(preferred_dir)
    dirs.extend([d for d in CANDIDATE_MASK_DIRS if d != preferred_dir])

    exts = []
    if preferred_ext: exts.append(preferred_ext)
    exts.extend([e for e in CANDIDATE_EXTS if e != preferred_ext])

    for d in dirs:
        for ext in exts:
            p = os.path.join(root_split_dir, d, stem + ext)
            if os.path.isfile(p):
                return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="VITON-HD root (e.g., DATA/VITON-HD)")
    ap.add_argument("--split", choices=["train","test"], default="train")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--mode", choices=["mask","maskfree"], default="mask",
                    help="'mask' -> 3 columns; 'maskfree' -> 2 columns")
    ap.add_argument("--mask_dirname", default=None, help="force a mask folder name (e.g., agnostic-mask)")
    ap.add_argument("--mask_ext", default=".png", help="preferred mask extension")
    ap.add_argument("--strict", action="store_true", help="error out if any file missing")
    args = ap.parse_args()

    pairs_txt = os.path.join(args.root, f"{args.split}_pairs.txt")
    split_dir = os.path.join(args.root, args.split)
    person_dir = os.path.join(split_dir, "image")
    cloth_dir  = os.path.join(split_dir, "cloth")

    assert os.path.isfile(pairs_txt), f"missing pairs: {pairs_txt}"
    assert os.path.isdir(person_dir), f"missing dir: {person_dir}"
    assert os.path.isdir(cloth_dir),  f"missing dir: {cloth_dir}"

    n_all = n_ok = 0
    with open(pairs_txt) as fin, open(args.out, "w", newline="") as fout:
        w = csv.writer(fout)
        for line in fin:
            line = line.strip()
            if not line: continue
            n_all += 1
            try:
                person_rel, cloth_rel = line.split()
            except ValueError:
                if args.strict: raise
                print(f"[warn] bad line: {line}"); continue

            person_path = os.path.join(person_dir, person_rel)
            cloth_path  = os.path.join(cloth_dir,  cloth_rel)
            if not (os.path.isfile(person_path) and os.path.isfile(cloth_path)):
                if args.strict: raise FileNotFoundError(f"missing person/cloth for: {line}")
                print(f"[warn] missing person/cloth: {line}")
                continue

            if args.mode == "maskfree":
                w.writerow([person_path, cloth_path])
                n_ok += 1
                continue

            mask_path = find_mask(split_dir, person_rel, args.mask_dirname, args.mask_ext)
            if mask_path is None:
                if args.strict: raise FileNotFoundError(f"mask not found for {person_rel}")
                print(f"[warn] mask not found for {person_rel}"); continue

            w.writerow([person_path, cloth_path, mask_path])
            n_ok += 1

    print(f"[done] wrote {args.out} ({n_ok}/{n_all} rows)")

if __name__ == "__main__":
    main()
