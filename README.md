![ezgif com-gif-maker (3)](https://user-images.githubusercontent.com/72849922/122694836-ad948680-d279-11eb-8d0d-3d9ea599b5f6.gif)

# FFmpeg_SuperResolution
![result](https://user-images.githubusercontent.com/72849922/122692663-67d3c000-d271-11eb-958c-27801a6c9bf7.PNG)



## WARNING
Maybe You can not operate this code in local server. I recommend you to operate this code in other server like A100, not local.


## Video

```bash
python save_video.py --weights_file "./{your weight file}.pth" \
                     --video_file "./{video file}.mp4" \
                     --model_name "{your model name}" \
                     --scale {2 or 3 or 4}
```


## Result
![ezgif com-gif-maker (3)](https://user-images.githubusercontent.com/72849922/122694845-b7b68500-d279-11eb-9977-70f31b87d069.gif)





