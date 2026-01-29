# 1. prepare structural data
python bin/fetch_structure_mapping.py --fasta-path data/train.fasta --pdb-dir data/pdb/ --output data/structure_index.json
python bin/make_structural_dataset.py --fasta-path data/train.fasta --structure-index data/structure_index.json --plm-cache-dir data/plm_cache/ --output data/structure_graphs/combined.pkl

# 2. bin/config USE_STRUCTURAL_ENCODER: True

# 3. train
python bin/main.py --train-path-name TRAIN_DATA_PATH --validation-path-name VAL_DATA_PATH ...