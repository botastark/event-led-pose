#!/usr/bin/env python3
"""Replace only the LED geometry in an OpenCV JSON/YAML pose config.

The 160 mm base and 100 mm height are from the supplied pose_calibrated.json.
Check these against the physical LED centers before treating pose as metric.
"""

import argparse
import ast
import json
import math
import re
from pathlib import Path


LED_POINTS_MM = {
    '165': [-80.0, 100.0, 0.0],
    '366': [80.0, 100.0, 0.0],
    '596': [0.0, 0.0, 0.0],
}


def report_previous_geometry(points: object) -> None:
    if isinstance(points, str):
        entries = re.findall(r'(?m)^\s+["\']?(165|366|596)["\']?\s*:\s*(\[[^\]]+\])', points)
        try:
            points = {hz: ast.literal_eval(values) for hz, values in entries}
        except (ValueError, SyntaxError):
            return
    if not isinstance(points, dict) or not all(hz in points for hz in LED_POINTS_MM):
        return
    try:
        a, b, tip = (points[hz] for hz in ('165', '366', '596'))
        width = math.dist(a, b)
        signed_height = 0.5 * (a[1] + b[1]) - tip[1]
    except (ValueError, TypeError, IndexError):
        return
    print(f'Previous base: {width:.2f} mm; vertical tip above base: '
          f'{signed_height:.2f} mm at zero rotation '
          '(negative means it projects below the base).')


def correct_config(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError('Choose a separate output path so the input remains available.')
    raw = source.read_text()
    if source.suffix.lower() == '.json':
        config = json.loads(raw)
        if 'led_points_mm' not in config or 'camera' not in config:
            raise ValueError('Expected led_points_mm and camera in the pose config.')
        old_points = config['led_points_mm']
        config['led_points_mm'] = LED_POINTS_MM
        destination.write_text(json.dumps(config, indent=2) + '\n')
    elif source.suffix.lower() in {'.yml', '.yaml'}:
        header = re.search(r'(?m)^led_points_mm:[ \t]*\r?\n', raw)
        if header is None:
            raise ValueError('Expected top-level led_points_mm: block.')
        following = re.search(r'(?m)^[^\s#%][^\n]*:', raw[header.end():])
        block_end = header.end() + following.start() if following else len(raw)
        old_points = raw[header.end():block_end]
        for hz in LED_POINTS_MM:
            if not re.search(rf'(?m)^\s+["\']?{hz}["\']?\s*:', old_points):
                raise ValueError(f'Expected LED {hz} in the existing YAML block.')
        replacement = ''.join(
            f'  "{hz}": [{point[0]:g}, {point[1]:g}, {point[2]:g}]\n'
            for hz, point in LED_POINTS_MM.items()
        )
        destination.write_text(raw[:header.end()] + replacement + raw[block_end:])
    else:
        raise ValueError('The pose config must be JSON, YML, or YAML.')
    print('Input LED coordinates:', old_points)
    report_previous_geometry(old_points)
    print('Output LED coordinates:', LED_POINTS_MM)
    print('Physical center distances: 165–366 = 160.0 mm; '
          '165–596 = 366–596 = %.2f mm' % math.hypot(80.0, 100.0))
    print('Wrote:', destination)
    print('Run with: --pose-config', destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Existing pose_config.json or .yml')
    parser.add_argument('output', type=Path, help='New pose config; existing file is preserved')
    args = parser.parse_args()
    correct_config(args.input, args.output)


if __name__ == '__main__':
    main()
