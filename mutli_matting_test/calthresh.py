
def calbyscore(path, thresh):
    sum = 0
    count = 0
    names = []
    pics = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        count += 1
        line = line.strip()
        factors = line.split(',')
        name = factors[0]
        ip = name.split('_')[0] + '_' + name.split('_')[1]
        if ip not in pics:
            pics.append(ip)
        label = int(factors[-1])
        score = float(factors[-2])
        if score >= thresh and label==1:
            if ip not in names:
                names.append(ip)
            sum += 1
    return sum/count, count, len(names)/len(pics), len(pics)

if __name__ == '__main__':
    projects = ['9_21', 'bg_test']
    epoch = 55
    threshs = [0.5, 0.6, 0.7, 0.8, 0.9]
    ious = [0.1, 0.2, 0.3, 0.4, 0.5]
    for project  in projects:
        for iou in ious:
            path = '/defaultShare/share/wujl/83/test/{}/txtpath/crop_{}_{}.txt'.format(project, iou, epoch)
            for thresh in threshs:
                boxnum, allbox, picnum, allpic = calbyscore(path, thresh)
                print('[*]! {}-{}-{} boxnum is :{}'.format(project, iou, thresh, boxnum))
                print('[*]! {}-{}-{} allbox is :{}'.format(project, iou, thresh, allbox))
                print('[*]! {}-{}-{} picnum is :{}'.format(project, iou, thresh, picnum))
                print('[*]! {}-{}-{} allpic is :{}'.format(project, iou, thresh, allpic))
                print('[*]*15')



