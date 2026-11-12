"""Canonical action parsing shared by environment wrappers."""
import re

def parse_command(command):
    match = re.fullmatch(r'\s*([a-z][a-z0-9_-]*)\(([^()]*)\)\s*', command)
    if not match:
        raise ValueError(f'Expected canonical functional action: {command!r}')
    args = tuple(part.strip() for part in match[2].split(',')) if match[2].strip() else ()
    return match[1], args
