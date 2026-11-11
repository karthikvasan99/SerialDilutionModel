import scipy.io as sio
print("train_data.mat:")
train = sio.loadmat('Data/train_data.mat')
for k, v in train.items():
    if not k.startswith('__'):
        try:
            print(f"{k}: {v.shape}")
        except:
            print(f"{k}: {type(v)}")

print("\ntest_data.mat:")
test = sio.loadmat('Data/test_data.mat')
for k, v in test.items():
    if not k.startswith('__'):
        try:
            print(f"{k}: {v.shape}")
        except:
            print(f"{k}: {type(v)}")
