from pathlib import Path

def fix_labels(d):
    fx = 0
    for lbl in Path(d).glob('*.txt'):
        text = lbl.read_text(encoding='utf-8', errors='ignore').strip()
        lines = text.splitlines()
        clean = []
        changed = False
        for l in lines:
            parts = l.strip().split()
            if not parts:
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                changed = True
                continue
            if cls > 1:
                changed = True
                continue
            # Normalize class id to int
            new_line = str(cls) + ' ' + ' '.join(parts[1:])
            if new_line != l:
                changed = True
            clean.append(new_line)
        if changed:
            lbl.write_text('\n'.join(clean) + ('\n' if clean else ''), encoding='utf-8')
            fx += 1
    print(f'Fixed {fx} in {d}')

fix_labels('data/3_processed/labels')
fix_labels('data/6_augmented/val/labels')
fix_labels('data/6_augmented/train/labels')
print('Done.')
