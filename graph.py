import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# df = pd.read_csv('20191027123042.csv', index_col=0)
df = pd.read_csv('20191027123244.csv', index_col=0)
for i in ['x', 'y', 'z']:
    df[i] = df[i] - df[i].shift(1)
# df = df.iloc[1:,:]
# print(df.to_string())
# print(len(df.index))

still = df[df['state'] == 0]
walk = df[df['state'] == 1]
run = df[df['state'] == 2]
slice = run.iloc[-10:-1,:]
print(slice.to_string())

chart = plt.figure(figsize=(10,10)).gca(projection='3d')
# chart.scatter(still['x'], still['z'], still['y'], marker = '.', c = 'blue')
# chart.scatter(walk['x'], walk['z'], walk['y'], marker = '.', c = 'red')
chart.scatter(run['x'], run['z'], run['y'], marker = '.', c = 'red')
chart.scatter(slice['x'], slice['z'], slice['y'], marker = 'o', c = 'blue')
chart.set_xlabel('x')
chart.set_ylabel('z')
chart.set_zlabel('y')
min, max = -0.1, 0.1
chart.set_xlim([min,max])
chart.set_ylim([min,max])
chart.set_zlim([min,max])
plt.show()
