import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.01,1,100)
y = [-np.log(1-i) for i in x]

plt.plot(x,y)
plt.title('- ln(1-x)')
plt.show()
