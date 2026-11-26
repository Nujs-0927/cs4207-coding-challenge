"""Package code, A's material and sample; omit raw/full-month data and runtimes."""
from pathlib import Path
import html
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parent
CHINESE = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')
TEXT_SUFFIXES = {'.py', '.html', '.txt', '.json', '.csv', '.sh', '.ipynb'}


def check_english_package_file(path):
    if CHINESE.search(str(path.relative_to(ROOT))):
        raise ValueError(f'Chinese characters in package filename: {path.name}')
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    text = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() in {'.json', '.ipynb'}:
        text = json.dumps(json.loads(text), ensure_ascii=False)
    if CHINESE.search(html.unescape(text)):
        raise ValueError(f'Chinese text in English-only package: {path.name}')


def main():
    verification = json.loads((ROOT / 'outputs/verification.json').read_text())
    if verification.get('status') != 'PASS':
        raise RuntimeError('Run verify_delivery.py successfully first')
    destination = ROOT / 'Group20_A_TEAM_HANDOFF.zip'
    files = []
    for p in ROOT.rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if p.suffix in ('.zip', '.pyc') or '__pycache__' in rel.parts:
            continue
        if rel.parts[0] == 'data' or 'cleaned_parts' in rel.parts or 'matplotlib_config' in rel.parts:
            continue
        check_english_package_file(p)
        files.append(p)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in sorted(files):
            archive.write(p, Path('Group20_ETA_A') / p.relative_to(ROOT))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP integrity check failed')
        names = archive.namelist()
        assert any(n.endswith('sample_100k.parquet') for n in names)
        assert not any('/data/raw/' in n or '/cleaned_parts/' in n for n in names)
    print(f'{destination}\n{len(files)} files; {destination.stat().st_size:,} bytes; integrity PASS')


if __name__ == '__main__':
    main()
