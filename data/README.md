# Data Setup

This project uses several public datasets. Because of their size and access
restrictions, raw data is **not stored in this repo** — everyone needs to
download it locally into `data/raw/`, which is gitignored.

## Folder structure

After setup, your local (not committed) folder should look like:

```
data/
└── raw/
    ├── compcars/
    │   ├── image/
    │   ├── label/
    │   ├── misc/
    │   ├── part/
    │   └── train_test_split/
    ├── stanford_cars/      (TBD)
    └── carDD/               (TBD)
```

## CompCars

CompCars requires requesting access directly from the dataset owners —
it isn't a direct download.

1. Go to the dataset page: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html
2. Fill out the access request form (usually requires a university email
   and stated research purpose — mention this is for a UTS capstone project
   with Dr Ali Haidar / NSW Police)
3. Once approved, you'll receive a download link (may be split into
   multiple parts / a password-protected archive — check the email
   carefully)
4. Extract the archive locally
5. Move/copy the extracted contents into `data/raw/compcars/` in your
   local repo, so the structure matches the layout above exactly
   (this matters — our EDA and scripts reference these exact paths)

**Note:** access approval can take a few days, so request it early rather
than waiting until you need the data.

**Note on CompCars colour labels:** the *web-nature* images (the main
`image/` folder) do **not** include colour labels — only the separate
*surveillance-nature* subset does. If you need colour data, check
whether your access grant includes the surveillance-nature portion
separately.

## Verifying your setup

Once your data is in place, run this from the repo root to sanity-check
the folder structure:

```bash
python -c "import os; print(os.listdir('data/raw/compcars'))"
```

You should see: `image`, `label`, `misc`, `part`, `train_test_split`

## A note on .gitignore

`data/raw/` and `data/processed/` are both gitignored — **never** try to
commit dataset files, even accidentally. If `git status` ever shows
hundreds/thousands of image files as untracked or staged, stop and check
`.gitignore` before committing anything.

If you're on Windows and need to edit `.gitignore`, do it directly in
VS Code's editor (not via PowerShell `echo`/redirection) — PowerShell
defaults to UTF-16 encoding, which git doesn't read correctly and will
silently break the ignore rules. Check the bottom-right corner of VS Code
shows "UTF-8" when editing this file.
