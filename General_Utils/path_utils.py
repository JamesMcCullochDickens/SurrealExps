import os, shutil
from pathlib import Path
import platform


def delete_all_files_in_folder(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def create_if_not_exists(dir_path, with_delete=False):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    else:
        if with_delete:
            delete_all_files_in_folder(dir_path)


def get_parent(path):
    path = Path(path)
    parent = path.parent.absolute()
    return str(parent)


def print_os_info():
    print(platform.platform())
    print(platform.system())
    print(platform.release())
    print()


def is_linux():
    return os.name == "posix"


def is_windows():
    return os.name == "nt"


def join_inner_paths(outer_dir, inner_fps=None):
    if not inner_fps:
        inner_fps = os.listdir(outer_dir)
    return [os.path.join(outer_dir, inner_fp) for inner_fp in inner_fps]


def get_first_elem_in_fold(outer_dir):
    inner_fps = os.listdir(outer_dir)
    return os.path.join(outer_dir, inner_fps[0])

