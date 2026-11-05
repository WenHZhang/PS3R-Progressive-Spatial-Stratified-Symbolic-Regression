# Publish the prepared release

The intended project name is `ps3r`; the prepared version is `0.2.0`.
The connected GitHub account found during preparation was `WenHZhang`.
PyPI's `ps3r` project JSON endpoint returned HTTP 404 during preparation.
An absent public project page does not reserve a name or guarantee that an
upload will be accepted.

## GitHub

The author created the public repository named `PS3R-Progressive-Spatial-Stratified-Symbolic-Regression` under `WenHZhang`. The package is uploaded while retaining the repository history
and its MIT license. The verified repository URL is:

https://github.com/WenHZhang/PS3R-Progressive-Spatial-Stratified-Symbolic-Regression

Use an authenticated clone of the existing repository so its initial history
is retained:

```bash
gh auth login --hostname github.com --git-protocol https --web --scopes workflow
gh auth setup-git
git clone https://github.com/WenHZhang/PS3R-Progressive-Spatial-Stratified-Symbolic-Regression.git
cd PS3R-Progressive-Spatial-Stratified-Symbolic-Regression
```

Copy the prepared source files into this clone, excluding `build/`, `dist/`,
Python caches, and generated checksum files. Review the changes, then publish:

```bash
git add .
git commit -m "Release ps3r 0.2.0"
git push -u origin main
```

The `.gitignore` keeps build caches and distributions out of source commits.
Create a GitHub release for `v0.2.0` and attach the checked wheel, source
distribution, and `SHA256SUMS.txt`. The CI workflow tests package installation,
resource integrity, imports, synthetic solver wiring, and bundled data.

## PyPI with GitHub Trusted Publishing

Sign in to the author's PyPI account. Before the first upload, add a pending
publisher at:

https://pypi.org/manage/account/publishing/

Use these values if publishing from the intended repository:

| Field | Value |
| --- | --- |
| PyPI project name | `ps3r` |
| GitHub owner | `WenHZhang` |
| GitHub repository | `PS3R-Progressive-Spatial-Stratified-Symbolic-Regression` |
| Workflow filename | `publish.yml` |
| Environment name | `pypi` |

After the repository checks pass, start **Publish to PyPI** from its Actions
page, or use the authenticated CLI:

```bash
gh workflow run publish.yml --ref main
```

The workflow is manually triggered; ordinary code pushes do not publish a
package. It requests a short-lived identity token;
no long-lived PyPI API token is stored in the repository. Creating a new
publisher changes account access and should be performed by the account owner.

Alternatively, the owner can upload the already checked files locally:

```bash
python -m pip install twine
python -m twine check --strict dist/*.whl dist/*.tar.gz
python -m twine upload dist/*.whl dist/*.tar.gz
```

Use the official credential prompt or a local credential store; do not put
API tokens in source files, command history, or chat messages.

## Verify publication

After a successful upload, inspect https://pypi.org/project/ps3r/0.2.0/ and
install that exact version in a new environment:

```bash
python -m pip install ps3r==0.2.0
python -c "import ps3r; from ps3r import PS3RModel; print(ps3r.__version__)"
ps3r validate all
```

Successful local builds are not evidence that either remote publication has
completed. Record the actual GitHub release URL and PyPI version URL after
their publication is observed.
