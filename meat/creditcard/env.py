import os
import sys


def touch():
    pass

def check_init(path: str) -> bool:
    if not os.path.exists(path) or path == '/':
        raise FileNotFoundError
    for filename in os.listdir(path):
        if filename == '__init__.py':
            return True
    return False

def get_parent_path(path: str) -> str:
    return os.path.abspath(os.path.dirname(path))

def set_module_path(path: str):
    sys.path.append(path)

# 1.获取当前真实绝对路径
FILE_PATH: str = os.path.abspath(os.path.realpath(__file__))
CURRENT_PATH: str = get_parent_path(FILE_PATH)
SEARCH_PATH: str = CURRENT_PATH
MODULE_PATH: str = CURRENT_PATH
# 2.check当前目录__init__.py,如果没则报错
res: str = check_init(SEARCH_PATH)
if not res:
    raise FileNotFoundError

# 3.如果有，check上一级路径，如果上级路径没有__init__.py,则设置当前路径到sys.path中
# 4.否则设置父路径为当前路径，重复2.3
# 5.知道当前路径为/，报错
while  True:
    SEARCH_PATH = get_parent_path(SEARCH_PATH)
    res = check_init(SEARCH_PATH)
    if res:
        continue
    else:
        set_module_path(SEARCH_PATH)
        break


if __name__ == '__main__':
    res: bool = check_init(CURRENT_PATH)
    print(res)
