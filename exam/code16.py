import torchaudio
import matplotlib.pyplot as plt

filename = r"0_0_0_0_1_1_1_1.wav"

#从语音文件读出音频图和采样率
waveform, sample_rate = torchaudio.load(filename)

#先查看当前的音频最大最小值，因为编码重构音频需要将输入的原音频数据规范到[-1,1]之间
print("Min of waveform:{}\nMax of waveform:{}\nMean of waveform:{}".format(waveform.min(),waveform.max(),waveform.mean()))

#如果音频数据本身就是在[-1,1]之间，就不需要进行缩放了
def normalize(tensor):
    # 减去均值，然后缩放到[-1,1]
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()

# waveform = normalize(waveform)


#先对音频数据进行编码，输入范围是[-1,1]，输出范围是[0,255]
transformed = torchaudio.transforms.MuLawEncoding()(waveform)
print("Shape of transformed waveform: {}".format(transformed.size()))
plt.figure()
plt.plot(transformed[0, :].numpy())
plt.show()

#然后对编码后的音频数据解码重构，输入范围是[0,255]，输出范围和原音频范围一致
reconstructed = torchaudio.transforms.MuLawDecoding()(transformed)
print("Shape of recovered waveform: {}".format(reconstructed.size()))
plt.figure()
plt.plot(reconstructed[0, :].numpy())
plt.show()

#最后将重构的音频数据和原始音频数据进行比较，查看误差率
err = ((waveform - reconstructed).abs() / waveform.abs()).median()
print("Median relative difference between original and MuLaw reconstucted signals: {:.2%}".format(err))