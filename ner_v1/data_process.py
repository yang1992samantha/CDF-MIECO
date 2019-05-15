file = open('./data/valid1.txt', 'r', encoding='utf-8')
file2 = open('./data/valid1.txt', 'w', encoding='utf-8')
for line in file.readlines():
    line = line.replace('\t', ' ')
    file2.write(line)

file2.close()
