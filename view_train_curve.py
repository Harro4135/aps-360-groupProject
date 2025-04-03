from numpy import genfromtxt
import matplotlib.pyplot as plt
from pathlib import Path

paths = Path("../stats/").glob("*.csv")
print(paths)
for path in paths:
    print(path)
    name = str(path).split('\\')[-1]
    # name = name.split('.')[0]
    print(name)
    my_data = genfromtxt(path, delimiter=',', skip_header=1)
    # print(my_data.shape)
    # print(my_data[0])
    plt.plot(my_data[:, 2], label='Training PCK')
    plt.plot(my_data[:, 1], label='Training Loss')
    plt.plot(my_data[:, 4], label='Validation PCK')
    plt.plot(my_data[:, 3], label='Validation Loss')
    plt.legend()
    plt.title(name)
    plt.xlabel('Epoch')
    # break
    plt.savefig('../res_imgs/' + name + '.png')
    plt.clf()