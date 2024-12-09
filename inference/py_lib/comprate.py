f=open('../result/result_1.txt')
lines=f.readlines()
rate_all=0
all=0
for line in lines:
    rate_all+=float(line.split(',')[0])*float(line.split(',')[1])
    all += float(line.split(',')[1])
print(rate_all/all)