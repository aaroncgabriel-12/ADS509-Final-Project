# Efficient upload

Use only the sibling `github_upload` folder as the repository root. A local Git
repository on branch `main` is initialized, with no commits and no remote URL.
No GitHub account or online repository was created.

1. In GitHub Desktop, choose **File > Add Local Repository**, then select
   `github_upload`. Do not select the submission parent or local archive.
2. Review the file list and included source data/notices. The large result ZIPs,
   weights, runtime caches and environment files should not be in the changes.
3. When satisfied, enter a first-commit message such as “Add final project code,
   saved results and evidence” and commit to main.
4. Choose Publish repository when you are ready and select the intended access.
   The course allows a private repository with instructor access. Public reuse
   requires reviewing the included data/license/provenance notices first.
5. Verify the GitHub page displays the root README, notebooks, inputs, results,
   evidence and docs. Arrange instructor access if private and copy its URL.

The file count exceeds GitHub's 100-file browser batch, so Desktop/Git is simpler
than several browser uploads. Individual payload files are all below the browser
25 MiB limit and Git's 100 MiB threshold; Git LFS is not needed for this package.
See the exact measured sizes in `../packaging/VALIDATION_REPORT.json`.
[Official GitHub upload guidance](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository).

The recorded presentation/video and Canvas submission are still pending. A local
folder, local Git initialization or an outline does not complete those steps.

The expanded phase is included in `expanded_evaluation/` and notebook 11. Its two small input ZIPs belong in the upload package. The full expanded result ZIP and original executed notebook remain in the sibling local archive. Current hashes cover both project phases.
