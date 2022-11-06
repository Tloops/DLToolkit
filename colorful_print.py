#   begin: \033[显示方式;字体色;背景色m
#     end: \0[33[0m
# --------------------------------------------
# 字体色     |       背景色     |      颜色描述
# --------------------------------------------
# 30        |        40       |       黑色
# 31        |        41       |       红色
# 32        |        42       |       绿色
# 33        |        43       |       黃色
# 34        |        44       |       蓝色
# 35        |        45       |       紫红色
# 36        |        46       |       青蓝色
# 37        |        47       |       白色
# --------------------------------------------
# --------------------------------
# 显示方式     |      效果
# --------------------------------
# 0           |     终端默认设置
# 1           |     高亮显示
# 4           |     使用下划线
# 5           |     闪烁
# 7           |     反白显示
# 8           |     不可见
# --------------------------------

redPrint = "\033[31m"
greenPrint = "\033[32m"
yellowPrint = "\033[33m"
bluePrint = "\033[34m"
purplePrint = "\033[35m"
cyanPrint = "\033[36m"
endPrint = "\033[0m"


def print_red(something, end='\n'):
    print(redPrint, something, endPrint, sep='', end=end)


def print_green(something, end='\n'):
    print(greenPrint, something, endPrint, sep='', end=end)


def print_yellow(something, end='\n'):
    print(yellowPrint, something, endPrint, sep='', end=end)


def print_blue(something, end='\n'):
    print(bluePrint, something, endPrint, sep='', end=end)


def print_purple(something, end='\n'):
    print(purplePrint, something, endPrint, sep='', end=end)


def print_cyan(something, end='\n'):
    print(cyanPrint, something, endPrint, sep='', end=end)
