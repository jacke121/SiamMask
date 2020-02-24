# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
import os
import time

from tools.test import *

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume',  type=str, default='../models/SiamMask_DAVIS.pth',
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='../experiments/siammask/config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default=r'D:\project\OctaveConv_pytorch\nn\track_pic', help='datasets')
args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # cam = cv2.VideoCapture(r"rtsp://admin:sbdwl123@192.168.25.42:554/h264/ch1/main/av_stream")
    cam = cv2.VideoCapture(0)

    index = 0

    ret, frame = cam.read()
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', frame, False, False)
        x, y, w, h = init_rect
    except Exception as e:
        print('selectROI',e)
        exit()

    toc = 0
    f=0
    while True:
        ret, frame = cam.read()
        tic = cv2.getTickCount()
        if f == 0:  # init
            start=time.time()
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(frame, target_pos, target_sz, siammask, cfg['hp'])  # init tracker
            print('init ok',time.time()-start)
        elif f > 0:  # tracking
            start=time.time()
            if frame is None:
                print("im is none")
                continue
            state = siamese_track(state, frame, mask_enable=True, refine_enable=True)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr
            score=state['score']
            score=score[score > 0.5]
            if len(score)>1:
                print('track time', len(score), time.time() - start)
                frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
                cv2.polylines(frame, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('SiamMask', frame)
            key = cv2.waitKey(1)
            # if key > 0:
            #     break
        f+=1

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
