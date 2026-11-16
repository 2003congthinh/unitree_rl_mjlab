import pandas as pd, glob
df = pd.read_parquet(glob.glob(r'..\bones-seed\metadata\*.parquet')[0])

for c in ['package', 'category', 'content_type_of_movement',
          'content_body_position', 'content_props', 'is_mirror', 'is_neutral']:
    print(f'\n== {c} ==')
    print(df[c].value_counts(dropna=False).head(15))

print('\nmean frames:', df.move_duration_frames.mean())