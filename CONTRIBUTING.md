# Contributing

Keep changes focused on a reproducible user-visible behavior. Start with an issue for new storage systems, polygon semantics or hosted model integrations, so the data contract and scope are clear before implementation.

Use `uv sync --locked --extra demo` for the core development environment, then run `uv run pytest`, `uv run ruff check .` and `uv run ruff format --check .`. For neural work, add `--extra neural` and separately run the actual-model experiment. Mock encoders establish contracts, not neural performance.

Preserve exact source passages and offsets, WGS84 coordinate order, hard geographic filtering and explicit generator labels. Include a regression for a bug fix. Describe executed checks and any unavailable coverage in pull requests. Do not commit credentials, private source records, generated indexes, model weights or virtual environments.

Demo data is authored in `data/build_demo.py`; regenerate `data/demo.geojson` after changing it. Evaluation IDs and gold document IDs must remain valid. Do not remove difficult or failed cases merely to improve the aggregate score.
