import torch
import torchaudio
import matplotlib.pyplot as plt

# filename = r"./waves_yesno/1_0_0_0_0_0_0_0.wav"
filename = r"./SpeechCommands/speech_commands_v0.02/cat/00b01445_nohash_0.wav"

#从语音文件读出音频图和采样率
waveform, sample_rate = torchaudio.load(filename)

print("Shape of waveform: {}".format(waveform.size()))#音频大小
#Shape of waveform: torch.Size([1, 53600])，单通道
print("Sample rate of waveform: {}".format(sample_rate))#采样率：8000
plt.plot(waveform.t().numpy())#画出波形图

'''
Spectrogram：从音频波形创建声谱图。

MuLawEncoding：基于mu-law比较编码波形。
MuLawDecode：解码mu-law编码波形。
AmplitudeToDB：将一个频谱图从频率/振幅刻度转换为分贝刻度，使用对数方法。
MelScale：使用转换矩阵将普通 STFT 转换为梅尔频率 STFT。

MelSpectrogram：使用 PyTorch 中的 STFT 函数从波形创建 MEL 频谱图。
MFCC：从音频波形创建梅尔频率的倒谱cepstrum系数。
'''


#传入音频图，转换成普通声谱图
specgram = torchaudio.transforms.Spectrogram()(waveform)
print("Shape of spectrogram: {}".format(specgram.size()))

#基于mu-law比较编码波形,将声谱图使用mu-law进行编码
specgram = torchaudio.transforms.MuLawEncoding()(specgram)
print("Shape of spectrogram: {}".format(specgram.size()))

#解码mu-law编码波形，将使用mu-law编码的结果进行解码输出
specgram = torchaudio.transforms.MuLawDecoding()(specgram)
print("Shape of spectrogram: {}".format(specgram.size()))


# 将一个频谱图/声谱图从功率/振幅刻度转换为分贝刻度，使用对数方法
specgram = torchaudio.transforms.AmplitudeToDB()(specgram)
print("Shape of spectrogram: {}".format(specgram.size()))

#将一个频谱图/声谱图使用转换矩阵转换为梅尔频率 FFT。
specgram = torchaudio.transforms.MelScale()(specgram)
print("Shape of spectrogram: {}".format(specgram.size()))


#传入音频图，转换成梅尔声谱图
specgram = torchaudio.transforms.MelSpectrogram()(waveform)
print("Shape of spectrogram: {}".format(specgram.size()))


#传入音频图，转换成梅尔频率倒谱系数频谱图
specgram = torchaudio.transforms.MFCC()(waveform)
print("Shape of spectrogram: {}".format(specgram.size()))

#使用对数压缩频谱图
plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')
plt.show()


