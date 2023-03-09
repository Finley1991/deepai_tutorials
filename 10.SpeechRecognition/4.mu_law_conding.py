import torch
import torchaudio
import matplotlib.pyplot as plt
filename = r"./waves_yesno/1_0_0_0_0_0_0_0.wav"
# filename = r"./SpeechCommands/speech_commands_v0.02/cat/00b01445_nohash_0.wav"

#从语音文件读出音频图和采样率
waveform, sample_rate = torchaudio.load(filename)

'''
对原音频图进行编码重构
对编码范围内小信号或大信号都采用等量化级进行量化 ,这对小信号来说是不利的。
为了提高小信号的利用，同时不降低对大信号的量化作用。也就是对小信号放大，对大信号量化压缩。
压缩量化的实质是“压大补小”,使小信号在整个动态范围内的信噪比基本一致。
由于小信号的幅度得到较大的放大 ,从而使小信号的信噪比大为改善。目前常用的压扩方法是对数型的 A律压缩和 μ律压缩
μ-law可以改善信噪比率而不需要增添更多的数据，使用mu-law算法来减少背景噪声。
在数字系统中，这个公式将数字信号压缩到8位，同时保持相同的噪音水平
μ律公式：y=ln(1+μx)/ln(1+μ）
其中 x 为归一化的量化器输入 , y 为归一化的量化器输出。常数 μ愈大 ,则小信号的压扩效益愈高 ,目前多采用 μ= 255
'''
#先查看当前的音频最大最小值，因为编码重构音频需要将输入的原音频数据规范到[-1,1]之间
print("Min of waveform:{}\nMax of waveform:{}\nMean of waveform:{}".format(waveform.min(),waveform.max(),waveform.mean()))

#如果音频数据本身就是在[-1,1]之间，就不需要进行缩放了
def normalize(tensor):
    # 减去均值，然后缩放到[-1,1]
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()

waveform = normalize(waveform)


#先对音频数据进行编码，输入范围是[-1,1]，输出范围是[0,255]
transformed = torchaudio.transforms.MuLawEncoding()(waveform)
print("Shape of transformed waveform: {}".format(transformed.size()))
plt.figure()
plt.plot(transformed[0, :].numpy())
plt.title("encoder")
plt.show()

#然后对编码后的音频数据解码重构，输入范围是[0,255]，输出范围和原音频范围一致
reconstructed = torchaudio.transforms.MuLawDecoding()(transformed)
print("Shape of recovered waveform: {}".format(reconstructed.size()))
plt.figure()
plt.plot(reconstructed[0, :].numpy())
plt.title("decoder")
plt.show()

#最后将重构的音频数据和原始音频数据进行比较，查看误差率
err_median = ((waveform - reconstructed).abs() / waveform.abs()).median()
err_mean = ((waveform - reconstructed).abs() / waveform.abs()).mean()
print("原始信号与Mu-Law重构信号的中值相对差: {:.2%}".format(err_median))
print("原始信号与Mu-Law重构信号的均值相对差: {:.2%}".format(err_mean))