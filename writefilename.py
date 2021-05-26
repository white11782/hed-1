import os

fileName = []
for file in os.listdir('E:\\2021-04\\HED\\hed_pytorch\\hed\\data\\TDP\\test\\'):
    fileName.append(file)

txtfile = open('E:\\2021-04\\HED\\hed_pytorch\\hed\\data\\TDP\\test.txt','a')
for line in fileName:
    txtfile.write('test/'+line+'\n')
txtfile.close()