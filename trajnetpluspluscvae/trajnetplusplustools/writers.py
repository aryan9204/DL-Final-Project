import json

from .data import SceneRow, TrackRow


def trajnet(row):
    if isinstance(row, SceneRow):
        payload = {
            'scene': {
                'id': row.scene,
                'p': row.pedestrian,
                's': row.start,
                'e': row.end,
                'fps': row.fps,
                'tag': row.tag,
            }
        }
        return json.dumps(payload)

    if isinstance(row, TrackRow):
        track = {
            'f': row.frame,
            'p': row.pedestrian,
            'x': row.x,
            'y': row.y,
        }
        if row.prediction_number is not None:
            track['prediction_number'] = row.prediction_number
        if row.scene_id is not None:
            track['scene_id'] = row.scene_id
        return json.dumps({'track': track})

    raise TypeError(f'Unsupported row type: {type(row)!r}')
