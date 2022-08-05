import csv


def extract_gr(path):
    csv_file = open('gt.csv', 'w', encoding='UTF8', newline='')
    writer = csv.writer(csv_file)
    with open(path, 'r') as f:
        lines = f.readlines()
    writer.writerow(['TIME', 'HR'])
    print('starting!')
    for line in lines:
        bpm_str, time_str = line.split(';')
        time = time_str[10:-1]
        hr = bpm_str[4:]
        writer.writerow([time, hr])
    csv_file.close()
    print('done!')


if __name__ == '__main__':
    extract_gr('../subject1-006-001-BPM.txt')

