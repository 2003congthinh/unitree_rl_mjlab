import pandas as pd
a = pd.read_csv(r'src\assets\motions\g1\dance1_subject2.csv', header=None, nrows=3)
b = pd.read_csv(r'src\assets\motions\g1\walk_ff_loop_180_R_003__A067.csv', header=None, nrows=3)
print('example:', a.shape); print(a.iloc[:, :8])
print('bones:  ', b.shape); print(b.iloc[:, :8])
print('bones total rows:', len(pd.read_csv(r'src\assets\motions\g1\walk_ff_loop_180_R_003__A067.csv', header=None)))
