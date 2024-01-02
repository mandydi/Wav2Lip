import scipy, os,cv2,sys, argparse, audio
import json, subprocess, random, string
import numpy as np
import torch, face_detection
from time import  strftime


# 读取音频的代码
def extract_audio_feature(audio_path,fps,save):
    mel_step_size = 16
    # 读取音频的代码
    if not audio_path.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, 'temp/temp.wav')
        subprocess.call(command, shell=True)
        audio_path = 'temp/temp.wav'

    wav = audio.load_wav(audio_path, 16000)
    # 提取Mel频谱特征
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1


    print("Length of mel chunks: {}".format(len(mel_chunks)))
    # 保存mel_chunks到指定路径
    np.save(os.path.join(save, 'mel_chunks.npy'), mel_chunks)

    return mel_chunks

def main():
    parser = argparse.ArgumentParser(
        description='extract_video_feature code to lip-sync videos in the wild using Wav2Lip models')
    parser.add_argument('--audio', type=str,
                        help='Filepath of video/audio file to use as raw audio source', required=True)
    parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                        default=25., required=False)
    parser.add_argument("--result_dir", default='./results', help="path to output")
    args = parser.parse_args()
    mel_step_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} for inference.'.format(device))

    # 创建保存文件夹
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    save_path = os.path.join(save_dir, 'extraction')
    os.makedirs(save_path, exist_ok=True)
    mel_martix=extract_audio_feature(args.audio,args.fps,save_path)
    print("shape of mel chunk:\n", mel_martix)

if __name__ == '__main__':
	main()