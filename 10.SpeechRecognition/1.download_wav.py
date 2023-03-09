import torchaudio,torch.utils.data
# dataset = torchaudio.datasets.YESNO('./', download=True)
dataset = torchaudio.datasets.SPEECHCOMMANDS('./', download=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

for x in data_loader:
    print(len(x))
    print(x[0].shape)
    print(x[1])
    print(x[2])
    print(x[3])
    print(x[4])