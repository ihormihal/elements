import csv

csvfile = open('data/data.csv')
data = csv.DictReader(csvfile, delimiter=',', quotechar='|')



# # with open('data/elements.csv') as csvfile:
# #     reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
# #     for row in reader:
# #         # print(''.join(row))
# #         print(row['symbol'])

for row in data:
    print(row['symbol'])