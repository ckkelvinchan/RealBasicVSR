# RealBasicVSR inference in mmcv2.0

This script was edited by [me](github.com/jere357) to run basicvsr inference on mmcv2.0 + mmagic since mmediting is deprecated now and there are some slight differences in the code of basicvsr in mmcv2.0 + mmagic vs the original code in mmediting. I have also added 2 tqdm progress bars (1 for nn inference and 1 work writing the video to the disk) so you can watch them fill up while your code is running.

* This code should work on your average mmcv2.0 + mmagic docker image i have also dumped the environment in the requirements.txt file so you can compare versions of packages and maybe locate the problem if you are having trouble running this code.


I have also included the updated_builder.py file which also needed very slight changes to work with mmcv2.0 + mmagic. For now this code only works and was tested in a <ins>video --> video</ins> setting. I have not tested it with images, that case might need some slight changes as well, I have also addedd .mkv as a readable format


* If you run into CUDA out of memory error try changing the seq_len parameter from None to something like 15

* feel free to contact me here or at jeronim@aimages.ai