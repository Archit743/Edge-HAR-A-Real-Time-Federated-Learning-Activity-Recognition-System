from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TEXT_EXTENSIONS = {'.csv', '.json', '.log', '.md', '.txt', '.toml'}
PHONE_PATH_RE = re.compile(r'(/storage/emulated/|/sdcard/|/data/data/|termux)', re.IGNORECASE)
RAW_SENSOR_RE = re.compile(r'(acc_x|acc_y|acc_z|gyro_x|gyro_y|gyro_z)', re.IGNORECASE)


@dataclass(slots=True)
class PrivacyScanResult:
    scanned_files: list[Path]
    violations: list[str]


def scan_artifacts(output_root: Path, report_path: Path) -> PrivacyScanResult:
    scanned: list[Path] = []
    violations: list[str] = []

    if not output_root.exists():
        return PrivacyScanResult(scanned_files=[], violations=['Output directory does not exist.'])

    for path in sorted(output_root.rglob('*')):
        if not path.is_file() or path == report_path or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        scanned.append(path)
        content = path.read_text(encoding='utf-8', errors='ignore')
        if PHONE_PATH_RE.search(content):
            violations.append(f'Phone-local path reference detected in {path}')
        if RAW_SENSOR_RE.search(content) and 'privacy_audit_report' not in path.name:
            violations.append(f'Potential raw sensor field leakage detected in {path}')

    return PrivacyScanResult(scanned_files=scanned, violations=violations)


def write_privacy_report(output_root: Path, report_path: Path) -> PrivacyScanResult:
    result = scan_artifacts(output_root=output_root, report_path=report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        '# Privacy Audit Report',
        '',
        '## Summary',
        f'- Files scanned: {len(result.scanned_files)}',
        f'- Violations found: {len(result.violations)}',
        '',
        '## Scanned Files',
    ]
    lines.extend(f'- {path}' for path in result.scanned_files or [Path('(none)')])
    lines.extend(['', '## Findings'])
    if result.violations:
        lines.extend(f'- {violation}' for violation in result.violations)
    else:
        lines.append('- No raw sensor data or phone-local path leakage detected in scanned artifacts.')

    report_path.write_text('\n'.join(str(line) for line in lines), encoding='utf-8')
    return result
