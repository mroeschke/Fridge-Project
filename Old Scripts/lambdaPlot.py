import numpy as np
import matplotlib.pyplot as plt

data = [[100, 80, 60, 40, 20, 0],
        [0.086336, 0.068264, 0.059624, 0.058825, 0.056632, 0.054329],
        [0.268242, 0.269987, 0.273306, 0.274517, 0.283817, 0.295530]]

plt.clf()
plt.style.use('ggplot')
fs = 12
fig, axis = plt.subplots(1,1)
fig.set_size_inches(6,3)
fs = 12
axis.plot(data[0], data[1], 'g-')
axis.set_ylabel('US$', color='g', fontsize=fs)
axis.set_xlabel('Lambda', fontsize=fs)
twin = axis.twinx()
twin.plot(data[0], data[2], 'r-')
twin.set_ylabel(r'lb CO$_2$', color='r', fontsize=fs)
plt.tight_layout()
plt.show()

fig.savefig('lambdaGraph.pdf')
