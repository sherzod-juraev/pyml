# Benchmarks and profiling

Tools that answer "where does the time go?" and "how much did a change move it?" with a
measurement instead of a guess. They are for the maintainers: nothing here is part of the package,
and nothing here runs in the CI.

| Question | Tool |
|:--|:--|
| Where does the time of this script go? | `run.py profile` (below) |
| How much faster or slower is this version than that one? | `asv` ([Compare versions with asv](#compare-versions-with-asv)) |

## Setup

```bash
uv sync --group dev-profile
```

This group installs `pyinstrument`. The plain `dev` group includes it too.

## Profile a script

```bash
python benchmarks/run.py profile examples/tree/decision-tree-classifier.py
```

The path can be any Python file: an example, a script of your own. Run the command by the
path of `run.py`, from any folder. Do not use `python -m benchmarks`: `-m` and `-c` put the
current folder first on `sys.path`, so from the repository root they could import the source
tree instead of the installed `pyml`.

| Option | Meaning |
|:--|:--|
| `--html` | Also write the interactive `pyinstrument` report to `benchmarks/results/<name>-pyinstrument.html`. |
| `--top N` | Rows in each `cProfile` table (default 20). |
| `--interval SECONDS` | Time between two `pyinstrument` samples (default 0.001). |

The script runs twice, once per profiler. They never run together, because each one slows
the other down and bends its numbers. The `matplotlib` backend is set to `Agg`, so examples do
not open windows.

## Reading the output

1. **`pyinstrument` tree.** The share of the wall-clock time in each call, from the script
   down to the function. It answers "where does the time go?". Start here.
2. **`cProfile` by cumulative time.** The same question from exact call counting, with the
   number of calls (`ncalls`) next to each function.
3. **`cProfile` by number of calls.** It answers "how often?". A function with a huge `ncalls`
   and a tiny time per call is overhead of the loop around it, not of the function itself.

Limits to keep in mind:

- `pyinstrument` samples, so a call shorter than the interval can be missed. A hot loop of tiny
  calls still shows up, because the samples add up.
- `cProfile` adds a cost to every call. Compare its shares and its counts, not its seconds with
  the seconds of `pyinstrument`.
- The first run pays for the imports (scipy, matplotlib), the second does not.
- Read the numbers of one machine against the numbers of the same machine. Do not copy numbers
  into issues as if they were facts about the library.

## Recipes

These are the two findings that started this folder, as they were at v0.5.1. After an
optimization the output changes, which is how to see that it worked.

**Loop overhead in the logistic models**

```bash
python benchmarks/run.py profile examples/linear-model/elastic-net-classifier.py
```

In the table by cumulative time, `_compute_loss`, `_compute_gradient` and `_predict_proba`
have the same `ncalls`, hundreds of thousands: one call per iteration of the training loop,
each taking tens of microseconds on arrays of a few kilobytes. The table by number of calls
shows the NumPy functions behind them (`sum`, `clip`, ...) with several calls per iteration.
The time is the overhead of many tiny NumPy calls, which in-place operations or `out=` do not
remove.

**Validation inside the loss of the regression models**

```bash
python benchmarks/run.py profile examples/linear-model/ridge.py
```

In the `pyinstrument` tree, `_compute_loss` is about half of `_fit`. It calls the public
`mean_squared_error`, and about half of that is `_validate_regression_inputs`. In the
`cProfile` table, `_validate_regression_inputs` has almost the same `ncalls` as `_compute_loss`:
the input is validated again on every iteration of the loop.

## Compare versions with asv

`asv` (airspeed velocity) builds `pyml` of any revision in its own virtual environment, runs the
suite of `benchmarks/asv/` against it and prints two revisions side by side. It is configured by
`asv.conf.json` in the repository root. Profile first to find *where* the time goes, then use
`asv` to show *how much* a change moved it.

### Setup

```bash
uv sync --group dev-bench
asv machine --yes
```

`asv machine --yes` runs once per computer. It stores the name and the hardware of this machine
in `~/.asv-machine.json` and accepts the defaults without asking. `asv` needs `uv` on the `PATH`,
because the configuration uses `uv` to create the environments.

### Compare two releases

```bash
asv run "v0.5.1^!"
asv run "v0.5.2^!"
asv compare v0.5.1 v0.5.2
```

- `<revision>^!` means "this commit only". Without it, `asv run v0.5.2` would measure every
  commit back to the start of the history. A tag, a branch name or a hash all work.
- The first `asv run` of a revision creates its environment, which takes a minute. A result is
  stored in `.asv/results/`, in the repository root (not in `benchmarks/`), so a revision is
  measured once and `asv compare` is instant.
- To compare your branch with `main`, commit first: `asv` builds a committed revision, not
  the files of the working tree. The suite itself is read from the working tree.

| Option of `asv compare` | Meaning |
|:--|:--|
| `--only-changed` | Hide the rows that did not change. |
| `--factor 1.2` | Call a row "changed" from 20% (default 1.1, that is 10%). |
| `--split` | Three tables: improved, same, worse. |
| `--sort ratio` | Sort by the size of the change. |

### Reading the table

- **Ratio** is `After / Before`. Below 1 is faster, above 1 is slower.
- **`±`** is the spread of the repeated measurements. A ratio smaller than the spread is noise.
- A `~` before a ratio means that `asv` found the difference not significant. Treat the two
  revisions as equal in that row.
- The first column marks a significant change: `-` improved, `+` got worse.
- The numbers belong to this machine. Do not run two revisions on a loaded computer, and never
  compare a number from one machine with a number from another.

### Add a benchmark

A benchmark is a class in a file `benchmarks/asv/bench_*.py`:

- `params` and `param_names` list the values to try (here, the number of samples).
- `setup(self, ...)` prepares the data. It is run before each measurement and is not timed.
- A method whose name starts with `time_` is timed.
- Fix the amount of work (`tol=0.0` with a fixed `max_iter` for the gradient descent models), so
  that two revisions do the same work and the ratio shows the cost of one step.
- Use only the public API of `pyml` that exists in the revisions you want to compare. `asv`
  reports a benchmark that raises as failed for that revision.

## Conventions

- `benchmarks/results/` holds the reports of `run.py profile`. It is ignored by git.
- `.asv/` in the repository root holds the environments, results and HTML report of `asv` (see
  `asv.conf.json`). It is ignored by git.
- No measured numbers are committed.
- A file in this folder must not have the name of a standard library module (`profile.py`,
  `random.py`, ...). Running `run.py` puts this folder first on `sys.path`, so such a file
  would hide the real module.
- The folder is checked by `ruff` and `mypy` (strict), like the rest of the repository.
