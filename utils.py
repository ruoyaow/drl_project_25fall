import io, json


def load_jsonl(file_path, mode="r", encoding="utf-8"):
    """
    Loads data from a .jsonl file, where each line is a 
    separate JSON object.
    """
    data = []
    # 'with open' is safer as it automatically closes the file
    with open(file_path, mode=mode, encoding=encoding) as f:
        for line in f:
            # Remove any leading/trailing whitespace (like newlines)
            line = line.strip()
            # Make sure we don't process empty lines
            if line:
                # Parse the single line of text as JSON
                data.append(json.loads(line))
    return data

def jload(f, mode="r"):
    if not isinstance(f, io.IOBase): f = open(f, mode=mode)
    jdict = json.load(f)
    f.close()
    return jdict

def jdump(obj, f, mode="w", indent=4, default=str):
    if not isinstance(f, io.IOBase): f = open(f, mode=mode)
    if isinstance(obj, (dict, list)): json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str): f.write(obj)
    else: raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()