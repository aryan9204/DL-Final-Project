import json
import random
from collections import defaultdict

import numpy as np

from .data import SceneRow, TrackRow


class Reader:
    def __init__(self, filename, scene_type='paths'):
        del scene_type
        self.filename = filename
        self.scenes_by_id = {}
        self._tracks_by_ped = defaultdict(list)
        self._prediction_tracks_by_scene = defaultdict(lambda: defaultdict(list))
        self._load()

    def _load(self):
        with open(self.filename) as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if 'scene' in item:
                    scene = item['scene']
                    self.scenes_by_id[scene['id']] = SceneRow(
                        scene=scene['id'],
                        pedestrian=scene['p'],
                        start=scene['s'],
                        end=scene['e'],
                        fps=scene['fps'],
                        tag=scene.get('tag', []),
                    )
                elif 'track' in item:
                    track = item['track']
                    row = TrackRow(
                        frame=track['f'],
                        pedestrian=track['p'],
                        x=track['x'],
                        y=track['y'],
                        prediction_number=track.get('prediction_number'),
                        scene_id=track.get('scene_id'),
                    )
                    if row.scene_id is None:
                        self._tracks_by_ped[row.pedestrian].append(row)
                    else:
                        self._prediction_tracks_by_scene[row.scene_id][row.pedestrian].append(row)

        for ped_id in self._tracks_by_ped:
            self._tracks_by_ped[ped_id].sort(key=lambda row: row.frame)
        for scene_id in self._prediction_tracks_by_scene:
            for ped_id in self._prediction_tracks_by_scene[scene_id]:
                self._prediction_tracks_by_scene[scene_id][ped_id].sort(
                    key=lambda row: ((row.prediction_number or 0), row.frame)
                )

    def scenes(self, sample=1.0, ids=None, limit=None, randomize=False):
        scene_ids = list(self.scenes_by_id.keys())
        if ids is not None:
            ids = set(ids)
            scene_ids = [scene_id for scene_id in scene_ids if scene_id in ids]
        scene_ids.sort()
        if randomize:
            random.shuffle(scene_ids)
        if sample < 1.0:
            keep = max(1, int(len(scene_ids) * sample))
            scene_ids = scene_ids[:keep]
        if limit is not None:
            scene_ids = scene_ids[:limit]

        for scene_id in scene_ids:
            yield scene_id, self.scene(scene_id)

    def scene(self, scene_id):
        if scene_id in self._prediction_tracks_by_scene:
            ped_tracks = self._prediction_tracks_by_scene[scene_id]
            ped_ids = sorted(ped_tracks.keys())
            primary_id = self.scenes_by_id[scene_id].pedestrian
            if primary_id in ped_ids:
                ped_ids.remove(primary_id)
                ped_ids.insert(0, primary_id)
            return [ped_tracks[ped_id] for ped_id in ped_ids]

        scene = self.scenes_by_id[scene_id]
        paths = []
        for ped_id, rows in self._tracks_by_ped.items():
            scene_rows = [row for row in rows if scene.start <= row.frame <= scene.end]
            if scene_rows:
                paths.append(scene_rows)

        paths.sort(key=lambda path: path[0].pedestrian)
        for idx, path in enumerate(paths):
            if path[0].pedestrian == scene.pedestrian:
                if idx != 0:
                    paths.insert(0, paths.pop(idx))
                break
        return paths

    @staticmethod
    def paths_to_xy(paths):
        frames = sorted({row.frame for path in paths for row in path})
        frame_to_idx = {frame: idx for idx, frame in enumerate(frames)}
        xy = np.full((len(frames), len(paths), 2), np.nan, dtype=np.float32)
        for path_idx, path in enumerate(paths):
            for row in path:
                frame_idx = frame_to_idx[row.frame]
                xy[frame_idx, path_idx, 0] = row.x
                xy[frame_idx, path_idx, 1] = row.y
        return xy
