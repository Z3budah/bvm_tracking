import scipy.io as scio
dataFile = '../lab/LSTM_training.mat'

data = scio.loadmat(dataFile)

print(data['DLC_sequences'].shape)

