import csv

# Number of coordinates of each hand
num_coords = 21

# Write columns name, each have x,y,z coordinate
landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]

# Write 'landmarks' list to a csv
with open('2hands.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)
    csv_writer.writerow('change hands')#kolom terakhir untuk membedakan perubahan gestur tangan
