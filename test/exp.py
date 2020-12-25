import os

def cal(dir):
    names = os.listdir(dir)
    repeat = []
    for name in names:
        id = name.split('_')[0] + '_' + name.split('_')[1]
        if id in repeat:
            continue
        else:
            repeat.append(id)
    return len(repeat)


dir = '/VisualGroup/share/wujl/83/test/9_21/croppic/croprecall_0.4_0.3_0.1'
print(cal(dir))