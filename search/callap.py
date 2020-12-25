


class CalData:
    def __init__(self, searchTxt, classiTxt, recall):
        self.searchTxt = searchTxt
        self.classiTxt = classiTxt
        self.recall = recall
        self.recallmaplist = [1, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 21]
        self.prcisionmaplist = [1, 3, 5, 6, 8, 12, 13, 14, 16, 18]

    def generateCdic(self):
        dic = {}
        with open(self.classiTxt, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            factors = line.split(',')
            label = int(factors[-1])
            name = factors[0]
            if self.recall:
                if label in self.recallmaplist:
                    dic[name] = 1
                else:
                    dic[name] = 0
            else:
                if label in self.prcisionmaplist:
                    dic[name] = 1
                else:
                    dic[name] = 0
        return dic

    def generateSdic(self):
        dic = {}
        with open(self.searchTxt, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            factors = line.split(',')
            name = factors[0]
            labels = [int(a) for a in factors[1:]]
            dic[name] = labels
        return dic

    def intrax(self):
        cdic = self.generateCdic()
        sdic = self.generateSdic()
        laps = [0]*6
        cats = [0]*6
        sum = 0
        for key in sdic.keys():
            sum += 1
            labels = sdic[key]
            tag = cdic[key]
            for i in range(6):
                if labels[i] == 1 and tag == 1:
                    laps[i] += 1
                if labels[i] == 1 or tag == 1:
                    cats[i] += 1
        print(sum)
        return laps, cats, [lap/sum for lap in laps], [cat/sum for cat in cats]


if __name__ == '__main__':
    projects = ['9_21', 'bg_test']
    for project in projects:
        if project == 'bg_test':
            recall = False
        else:
            recall = True
        searchtxtra = '/defaultShare/share/wujl/83/test/{}/croppic/crop_0.1/featurea.txt'.format(project)
        searchtxtrs = '/defaultShare/share/wujl/83/test/{}/croppic/crop_0.1/features.txt'.format(project)
        classtxt = '/defaultShare/share/wujl/83/test/{}/txtpath/crop_0.4_50_single.txt'.format(project)
        print(project)
        cal1 = CalData(searchtxtra, classtxt, recall)
        laps, cats, lapsrate, catsrate = cal1.intrax()
        print('a')
        print(lapsrate)
        print(catsrate)
        cal2 = CalData(searchtxtrs, classtxt, recall)
        laps1, cats1, lapsrate, catsrate = cal2.intrax()
        print('s')
        print(lapsrate)
        print(catsrate)
