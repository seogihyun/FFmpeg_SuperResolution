
# FFmpeg_SuperResolution
![result](https://user-images.githubusercontent.com/72849922/122692663-67d3c000-d271-11eb-958c-27801a6c9bf7.PNG)



## WARNING
Maybe You can not operate this code in local server. I recommand you try to operate this code in other server like A100, not local.


## Video

```bash
python save_video.py --weights_file "./{your weight file}.pth" \
                     --video_file "./{video file}.mp4" \
                     --model_name "{your model name}" \
                     --scale {2 or 3 or 4}
```


## Result
![opencv_fsrcnn-x](https://user-images.githubusercontent.com/72849922/122024681-9ed64b80-ce03-11eb-8564-102d6fd693ba.gif)




