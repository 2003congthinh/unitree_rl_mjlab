import pandas as pd, glob
df = pd.read_parquet(glob.glob(r'..\bones-seed\metadata\*.parquet')[0])

sel = df[
    (df.is_mirror.astype(str).str.lower() == 'false')          # works for bool or string
    & (df.content_props.astype(str) == '0')                    # 0 = no props
    & (df.content_body_position == 'standing')
    & (df.is_neutral == 1)                                      # neutral style, no acting
    & (df.move_duration_frames.between(360, 720))              # 3-6 s at 120 fps
    & (df.content_type_of_movement.isin(['walking', 'gesture', 'standing idle']))
]
print('matches:', len(sel))

cols = ['move_name', 'move_duration_frames', 'content_type_of_movement',
        'content_short_description', 'move_g1_path']
print(sel[cols].sample(min(20, len(sel)), random_state=0).to_string())