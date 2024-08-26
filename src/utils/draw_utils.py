import numpy as np
import cv2
import sys
from src.utils import get_center_of_bbox, get_bbox_width


def draw_ellipse(frame, bbox, color, track_id=None):
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width), int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4
    )

    rectangle_width = 40
    rectangle_height = 20
    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    y1_rect = (y2 - rectangle_height // 2) + 15
    y2_rect = (y2 + rectangle_height // 2) + 15

    if track_id is not None:
        cv2.rectangle(
            frame,
            (int(x1_rect), int(y1_rect)),
            (int(x2_rect), int(y2_rect)),
            color,
            cv2.FILLED
        )

        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10

        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text), int(y1_rect + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return frame


def draw_triangle(frame, bbox, color):
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x, y],
        [x - 10, y - 20],
        [x + 10, y - 20],
    ])
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    return frame


def draw_annotations(video_frames, tracks):
    output_video_frames = []
    for frame_num, frame in enumerate(video_frames):
        frame = frame.copy()

        player_dict = tracks['players'][frame_num]
        ball_dict = tracks['ball'][frame_num]
        referee_dict = tracks['referees'][frame_num]

        # Draw Players
        for track_id, player in player_dict.items():
            color = player.get('team_color', (0, 0, 255))
            frame = draw_ellipse(frame, player['bbox'], color, track_id)

        # Draw Referees
        for _, referee in referee_dict.items():
            frame = draw_ellipse(frame, referee['bbox'], (0, 255, 255))

        # Draw Ball
        for track_id, ball in ball_dict.items():
            frame = draw_triangle(frame, ball['bbox'], (0, 255, 0))

        output_video_frames.append(frame)

    return output_video_frames
