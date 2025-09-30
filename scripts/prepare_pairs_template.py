
import csv, argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    rows = [
        {'patient_id': 'P001', 'side': 'R', 'cc_path': '/path/to/P001_CC.png', 'mlo_path': '/path/to/P001_MLO.png', 'label': 0},
        {'patient_id': 'P002', 'side': 'L', 'cc_path': '/path/to/P002_CC.png', 'mlo_path': '/path/to/P002_MLO.png', 'label': 2},
    ]
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['patient_id','side','cc_path','mlo_path','label'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote template CSV to {args.out}")
if __name__ == "__main__":
    main()
