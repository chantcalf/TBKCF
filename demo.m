video_path = 'sequences/Jogging';
[seq, ground_truth] = load_video_info(video_path);

% Run ECO
results = run_TBKCF(seq,'',0);