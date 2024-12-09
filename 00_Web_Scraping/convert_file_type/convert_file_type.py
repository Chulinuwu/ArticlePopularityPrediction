import os

def rename_files_to_json(folder_path):
    for filename in os.listdir(folder_path):
        base_file, ext = os.path.splitext(filename)
        new_filename = base_file + '.json'
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

if __name__ == "__main__" :
    parent_folder_path = 'folder_path' # Change folder_path
    for folder_name in os.listdir(parent_folder_path):
        folder_path = os.path.join(parent_folder_path, folder_name)
        if os.path.isdir(folder_path):
            rename_files_to_json(folder_path)