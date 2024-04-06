import json


def write_readable_cached_dict(file_path:str, d:dict) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=4)

