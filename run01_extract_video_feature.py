import scipy, cv2, os, sys, argparse, audio
import numpy as np
from scipy.io import loadmat, savemat
#源程序是用保存位mat格式
from time import  strftime
import torch, face_detection
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

#读取视频帧、剪切、旋转
def frame_spilt(face,fps,resize_factor,crop,rotate):
    # 读取视频并对视频帧进行处理的代码
    if not os.path.isfile(face):
        raise ValueError('--face argument must be a valid path to video/image file')
    elif face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(face)]
        tmp_fps = fps
    else:
        video_stream = cv2.VideoCapture(face)
        tmp_fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')
        print("fps is:", tmp_fps)

        # 对每一帧进行面部检测
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))

            if rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    return full_frames


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)
    batch_size = args.face_det_batch_size

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results



def main(args):
    # 创建保存文件夹
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    first_frame_dir = os.path.join(save_dir, 'extraction')
    os.makedirs(first_frame_dir, exist_ok=True)
    save_path = os.path.join(first_frame_dir, 'full_frame.npy')
    save_path2 = os.path.join(first_frame_dir, 'video_extraction.npy')
    full_frames=frame_spilt(args.face,args.fps,args.resize_factor,args.crop,args.rotate)

    #打印帧数量
    print("Number of frames available for inference: " + str(len(full_frames)))
    #face_det_results包含了每一帧图像中检测到的人脸区域的图像和位置信息
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(full_frames)  # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([full_frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in full_frames]

    # 以二进制格式进行保存
    full_frames_array = np.array(full_frames)
    np.save(save_path, full_frames_array)
    face_det_array=np.array(face_det_results)
    np.save(save_path2,face_det_array)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='extract_video_feature code to lip-sync videos in the wild using Wav2Lip models')
    parser.add_argument('--face', type=str,
                        help='Filepath of video/image that contains faces to use', required=True)
    parser.add_argument('--resize_factor', default=1, type=int,
                        help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')
    parser.add_argument('--rotate', default=False, action='store_true',
                        help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                             'Use if you get a flipped result, despite feeding a normal looking video')
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                             'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')
    args = parser.parse_args()
    main(args)

