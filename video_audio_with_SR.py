import ffmpeg
import cv2
import torch
import argparse
import numpy as np
import os.path
import time
import torch.backends.cudnn as cudnn
import lpips as lp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from models import FSRCNN_x
from utils import preprocess, calc_avg



def writeVideo(args):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    """ video file 경로를 가져와서 이름과 확장자 명으로 나눔 """
    path = args.video_file
    name, extension = os.path.basename(path).split('.')

    """ video 파일을 가져와서 capture """
    try:
        cap = cv2.VideoCapture('./{title}.{extension}'.format(title=str(name), extension=str(extension)))
 
    except:
        print('Try again!')
        return

    """ capture한 이미지의 width와 height 값을 가져옴"""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ##################################################
    
    """ Process Pipeline setting """
    process1 = (
        ffmpeg
        .input('./{title}.{extension}'.format(title=str(name), extension=str(extension)))
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True) # if True, connect pipe to subprocess stdout 
    )

    process2 = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output('{name}_{model_name}_{title}.{extension}'.format(name='Gihyun', model_name=args.model_name, title=str(name), extension=str(extension)), pix_fmt='yuv420p')
        .overwrite_output()
        .run_async(pipe_stdin=True) # if True, connect pipe to subprocess stdin
    )
    ##################################################
    
    """ model load """
    model = FSRCNN_x(scale_factor=args.scale).to(device)

    """ 학습시킨 weighs와 parameters가 들어있는 파일을 가져와 model에 저장 """
    try:
        model.load_state_dict(torch.load(args.weights_file, map_location=device))
    except:
        state_dict = model.state_dict()
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
    
    model.eval()
    
    """ LPIPS 계산 """
    lpips_metric = lp.LPIPS(net='vgg')

    while True:
        
        """ process1 pipeline을 통과한 이미지를 불러와 byte값으로 저장 ?? 맞나?? """ 
        in_bytes = process1.stdout.read(width * height * 3)

        if in_bytes:
            """ height, width, channel=3으로 된 frame 생성 """
            in_frame = (
                np
                .frombuffer(in_bytes, np.uint8) # np.frombuffer(buffer, dtype=float, count=-1, offset=0, *, like=None) -- buffer : An object that exposes the buffer interface.
                .reshape([height, width, 3])
            )      

            ############## Super Resolution #####################
            """ Image size 조절 """
            hr = cv2.resize(in_frame, (width, height), interpolation=cv2.INTER_CUBIC)
            lr = cv2.resize(hr, (width // args.scale, height // args.scale), interpolation=cv2.INTER_CUBIC)
            bicubic = cv2.resize(lr, (width ,height), interpolation=cv2.INTER_CUBIC)

            # ycbcr값
            lr = preprocess(lr, device)
            hr = preprocess(hr, device)

            with torch.no_grad():
                preds = model(lr).clamp(0.0, 1.0)
            
            """ PSNR, SSIM, LPIPS 평균값 구하기 """
            if args.calc_check:

                psnr_avg, _, _ = calc_avg(hr, preds, lpips_metric)
                _, ssim_avg, _ = calc_avg(hr, preds, lpips_metric)
                _, _, lpips_avg = calc_avg(hr, preds, lpips_metric)

            preds = preds.mul(255.0).cpu().numpy().squeeze(0)

            """ output : (c,h,w) -> (h,w,c)로 변경 """
            sr_image = np.array(preds).transpose([1, 2, 0])
            sr_image = np.clip(sr_image, 0.0, 255.0).astype(np.uint8).copy() 
            
            ####################################################

            """ 이미지 중앙 부분 절반씩 잘라서 붙이기"""
            bicubic = bicubic[:, int(width*(1/4)):int(width*(3/4))]
            sr_image = sr_image[:, int(width*(1/4)):int(width*(3/4))]

            """ 이미지에 Text 넣기 """
            bicubic_name = 'Bicubic'
            SR_name = 'FSRCNN-x'
            org = (450, 50) # org (글자 위치)
            font = cv2.FONT_HERSHEY_SIMPLEX  
            fontScale = 1
            color = (0, 0, 0)  
            thickness = 2 # Line thickness of 2 px (두께) 
            # Using cv2.putText() method 
            bicubic = cv2.putText(bicubic, bicubic_name, org, font, fontScale, color, thickness, cv2.LINE_AA)
            # Using cv2.putText() method 
            sr_image = cv2.putText(sr_image, SR_name, org, font, fontScale, color, thickness, cv2.LINE_AA)   

            """ 잘라서 나눴던 이미지 합치기 """
            output = cv2.hconcat([bicubic, sr_image])

            """ process2 pipeline 통과 """
            process2.stdin.write(
                output
                .astype(np.uint8)
                .tobytes()
            )
        else :
            break

        if args.calc_check:
            """ PSNR, SSIM, LPIPS 평균 """
            print('PSNR: {:.2f}'.format(psnr_avg))
            print('SSIM: {:.2f}'.format(ssim_avg))
            print('LPIPS: {:.2f}'.format(lpips_avg))

    """ process pipeline 종료 """
    process2.stdin.close()
    process1.wait()
    process2.wait()

    """ 영상과 오디오 합치기 """
    original_video = ffmpeg.input('./{title}.{extension}'.format(title=str(name), extension=str(extension)))
    sr_video = ffmpeg.input('./{name}_{model_name}_{title}.{extension}'.format(name='Gihyun', model_name=args.model_name, title=str(name), extension=str(extension)))
    audios = original_video.audio
    videos = sr_video.video
    join = ffmpeg.output(videos, audios,'./{name}_{model_name}_merged_{title}.{extension}'.format(name='Gihyun', model_name=args.model_name, title=str(name), extension=str(extension)))
    join.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_file', type=str, required=True)
    parser.add_argument('--video_file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='FSRCNN-x')
    parser.add_argument('--calc_check', action='store_true')
    args = parser.parse_args()
    writeVideo(args)
