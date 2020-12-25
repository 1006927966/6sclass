def calbscore(txtpath, thresh, maplist):
    sum = 0
    count = 0
    names = []
    pics = []
    with open(txtpath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        count += 1
        line = line.strip()
        factors = line.split(',')
        label = int(factors[-1])
        score = float(factors[-2])
        name = factors[0]
        ip = name.split('_')[0] + '_' + name.split('_')[1]
        if ip not in pics:
            pics.append(ip)
        if label in maplist and score>thresh:
            sum += 1
            if ip not in names:
                names.append(ip)
    return sum/count, count, len(names)/len(pics), len(pics)

if __name__ == '__main__':
    projects = ['9_21', 'bg_test']
    epoch = 50
    threshs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ious = [0.1, 0.2, 0.3, 0.4, 0.5]
    recalllist = [1, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 21]
    alertlist = [1, 3, 5, 6, 8, 12, 13, 14, 16, 18]
    for project  in projects:
        if project == '9_21':
            maplist = recalllist
        else:
            maplist = alertlist
        for iou in ious:
            path = '/defaultShare/share/wujl/83/test/{}/txtpath/crop_{}_{}_single.txt'.format(project, iou, epoch)
            for thresh in threshs:
                boxnum, allbox, picnum, allpic = calbscore(path, thresh, maplist)
                print('[*]! {}-{}-{} boxnum is :{}'.format(project, iou, thresh, boxnum))
                print('[*]! {}-{}-{} allbox is :{}'.format(project, iou, thresh, allbox))
                print('[*]! {}-{}-{} picnum is :{}'.format(project, iou, thresh, picnum))
                print('[*]! {}-{}-{} allpic is :{}'.format(project, iou, thresh, allpic))
                print('[*]*15')


