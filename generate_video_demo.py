import glob

import cv2
import mmcv
import numpy as np


class VideoDemo:
    ''' Generate video demo given two sets of images.

    Please note that there will be video compression when you save the output
    as a video. Therefore, the result would be inferior to the actual outputs.

    Args:
        input_left_dir (str): The directory storing the images at the left.
        input_right_dir (str): The directory storing the images at the right.
        output_path (str): The path of the output video file.
        start_frame (int): The first frame to start sliding.
        pause_frame (int): The frame where a pause is raised.
        repeat_when_pause (int): The number of frames to be repeated when
            paused.
        slide_step (int): The step size of each slide. It controls the speed of
            the slide.
        line_width (int): The width of the line separating two images.
        frame_rate (int): The frame rate of the output video.

    '''

    def __init__(self, input_left_dir, input_right_dir, output_path,
                 start_frame, pause_frame, repeat_when_pause, slide_step,
                 line_width, frame_rate):

        self.paths_left = sorted(glob.glob(f'{input_left_dir}/*'))
        self.paths_right = sorted(glob.glob(f'{input_right_dir}/*'))

        self.output_path = output_path
        self.start_frame = start_frame
        self.pause_frame = pause_frame
        self.repeat_when_pause = repeat_when_pause
        self.slide_step = slide_step
        self.line_width = line_width
        self.frame_rate = frame_rate

        # initialize video writer
        self.video_writer = None

    def merge_images(self, img_left, img_right, x_coord):
        img_out = np.copy(img_left)
        img_out[:, x_coord:, :] = img_right[:, x_coord:, :]

        # add white line
        img_out[:, x_coord:x_coord + self.line_width, :] *= 0
        img_out[:, x_coord:x_coord + self.line_width, :] += 255

        return img_out

    def __call__(self):
        for i, (path_left, path_right) in enumerate(
                zip(self.paths_left, self.paths_right)):

            # start sliding
            if i >= self.start_frame:
                img_left = mmcv.imread(path_left, backend='cv2')
                img_right = mmcv.imread(path_right, backend='cv2')
                # img_right = mmcv.imrescale(
                #     img_right, 4, interpolation='nearest', backend='cv2')
                current_idx = self.slide_step * (i - self.start_frame)
                img_out = self.merge_images(img_left, img_right, current_idx)

            else:
                img_out = mmcv.imread(path_left, backend='cv2')

            # create video writer if haven't
            if self.video_writer is None:
                h, w = img_out.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.output_path, fourcc,
                                                    self.frame_rate, (w, h))

            self.video_writer.write(img_out.astype(np.uint8))

            # pause at somewhere
            if i == self.pause_frame:
                for _ in range(0, self.repeat_when_pause):
                    self.video_writer.write(img_out.astype(np.uint8))

        # pause before sliding over the last frame
        for _ in range(0, self.repeat_when_pause):
            self.video_writer.write(img_out.astype(np.uint8))

        # slide over the last frame
        w = img_out.shape[1]
        current_idx = min(current_idx, w - self.line_width)
        while current_idx + self.line_width >= 0:
            img_out = self.merge_images(img_left, img_right, current_idx)
            self.video_writer.write(img_out.astype(np.uint8))

            current_idx -= self.slide_step

        # pause before ending the demo
        self.video_writer.write(img_right.astype(np.uint8))
        for _ in range(0, self.repeat_when_pause):
            self.video_writer.write(img_right.astype(np.uint8))

        cv2.destroyAllWindows()
        self.video_writer.release()


if __name__ == '__main__':
    """
    Assuming you have used our demo code to generate output images in
    results/demo_000. You can then use the following code to generate a video
    demo.
    """

    video_demo = VideoDemo(
        input_left_dir='results/demo_000',
        input_right_dir='data/demo_000',
        output_path='demo_video.mp4',
        start_frame=5,
        pause_frame=15,
        repeat_when_pause=25,
        slide_step=100,
        line_width=10,
        frame_rate=25,
    )
    video_demo()
