python make_vitonhd_csv.py \
  --root DATA/VITON-HD \
  --split train \
  --out vitonhd_train_mask.csv \
  --mode mask \
  --mask_dirname agnostic-mask   # 데이터에 따라 agnostic-v3.2 로 바꿔도 됨