import torch
import torchaudio
import matplotlib.pyplot as plt

# filename = r"./waves_yesno/1_0_0_0_0_0_0_0.wav"
filename = r"./SpeechCommands/speech_commands_v0.02/cat/00b01445_nohash_0.wav"

#从语音文件读出音频图和采样率
waveform, sample_rate = torchaudio.load(filename)

print("Shape of waveform: {}".format(waveform.size()))#音频大小
#Shape of waveform: torch.Size([1, 16000])，单通道
print("Sample rate of waveform: {}".format(sample_rate))#采样率：16000
plt.figure()
plt.plot(waveform.t().numpy())
plt.title("raw")
# plt.show()
# exit()

'''
对原图从采样，改变原有的采样率
'''
#采样率缩小10倍，每秒采到的数据变少10倍，采集的数据更加粗糙
new_sample_rate = sample_rate/10 #1600
channel = 0
transformed = torchaudio.transforms.Resample(sample_rate,new_sample_rate)(waveform.view(1,-1))
print("Shape of transformed waveform:{}".format(transformed.size()))
plt.figure()
plt.plot(transformed.t().numpy())
plt.title("new")
plt.show()
