import yaml

def load_yaml_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data
