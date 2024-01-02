import torch, cv2, os, sys, argparse, audio
import numpy as np
from tqdm import tqdm
from models import Wav2Lip
from os.path import splitext
from pathlib import Path
import subprocess
import torch, face_detection
from run01_extract_video_feature import face_detect

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    print("Loading checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    if torch.cuda.is_available():
        model = model.cuda()
    return model.eval()

def datagen(frames, mels,face_det_results):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


def main(args):
    if not os.path.isfile(args.face) or not args.face.endswith('.npy'):
        raise ValueError('--face argument must be a valid path to .npy file')

    if not os.path.isfile(args.audio) or not args.audio.endswith('.npy'):
        raise ValueError('--audio argument must be a valid path to .npy file')

    mel_chunks = np.load(args.audio)
    full_frames = np.load(args.face)
    face_detect_info = np.load(args.face_detect)

    model = load_model(args.checkpoint_path)
    print("Model loaded")

    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks,face_detect_info.copy())

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
        if i == 0:
            model = load_model(args.checkpoint_path)
            print("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi',
                                  cv2.VideoWriter_fourcc(*'DIVX'), args.fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()

    output_file = splitext(args.outfile)[0] + '.mp4'
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio_path, 'temp/result.avi', output_file)
    subprocess.call(command, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lip sync videos based on audio .npy and face .npy files')

    parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from',
                        required=True)

    parser.add_argument('--face', type=str, help='Processed .npy file of video', required=True)

    parser.add_argument('--audio', type=str, help='Processed .npy file of audio', required=True)

    parser.add_argument('--audio_path', type=str, help='Original audio path', required=True)
    parser.add_argument('--face_detection_path', type=str, help='Original face_detect_info path', required=True)
    parser.add_argument('--outfile', type=str, help='Video path to save result',
                        default='results/result_test.mp4')
    parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                        default=25., required=False)
    parser.add_argument('--resize_factor', default=1, type=int,
                        help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')
    parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)
    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                             'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')
    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                             'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')
    parser.add_argument('--static', type=bool,
                        help='If True, then use only first video frame for inference', default=False)
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                        help='Padding (top, bottom, left, right). Please adjust to include chin at least')
    parser.add_argument('--face_det_batch_size', type=int,
                        help='Batch size for face detection', default=16)
    parser.add_argument('--rotate', default=False, action='store_true',
                        help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                             'Use if you get a flipped result, despite feeding a normal looking video')
    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')
    args = parser.parse_args()
    args.img_size = 96
    main(args)