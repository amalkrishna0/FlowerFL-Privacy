def read_data(file_path):
    with open(file_path, "rb") as f:
        return f.read()

def write_data(file_path, data):
    with open(file_path, "wb") as f:
        f.write(data)
