from utils import read_video, save_video, draw_annotations, draw_camera_movement, draw_speed_and_distance
from trackers import Tracker, interpolate_ball_positions, add_position_to_tracks
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator, add_adjust_positions_to_tracks
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator


def main():
    video_name = '08fd33_4'
    # Read Video
    video_frames, fps = read_video('data/input_videos/' + video_name + '.mp4')

    # Initialize Tracker
    tracker = Tracker('data/models/v8/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='data/stubs/track_stubs_ ' + video_name + '.pkl')

    # Get object positions
    add_position_to_tracks(tracks)

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='data/stubs/camera_movement' +
                                                                                        video_name + '.pkl')
    add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks['ball'] = interpolate_ball_positions(tracks['ball'])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator(fps)
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball to Player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            try:
                team_ball_control.append(team_ball_control[-1])
            except IndexError:
                pass

    team_ball_control = np.array(team_ball_control)

    # Draw output
    # Draw object Tracks
    output_video_frames = draw_annotations(video_frames, tracks, team_ball_control)

    # Draw camera movement
    output_video_frames = draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Draw Speed and Distance
    draw_speed_and_distance(output_video_frames, tracks)

    # Save Video
    save_video(output_video_frames, 'data/output_videos/output_' + video_name + '.avi', fps)


if __name__ == '__main__':
    main()
