import numpy as np

success = []
for i in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]:
    file = str(i) + '_time.txt'
    data = np.loadtxt(file)
    success.append(np.sum(data < 1000))
print(success)
