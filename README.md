#### install uv for project manangement
`brew install uv ffmpeg`

#### install all the python deps for project
`uv sync`

#### run the format transformer
`uv run nio_to_rerun.py data.zip output.rrd --max-frames 20`

#### run the rerun viewer
`uv run rerun output.rrd`
