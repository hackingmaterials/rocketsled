# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import os
import json
import webbrowser
import requests
import datetime
from invoke import task

from rocketsled import __version__
from monty.os import cd

"""
Deployment file to facilitate releases.
"""


@task
def make_doc(ctx):
    with cd("docs_rst"):
        ctx.run("sphinx-apidoc -o . -f ../rocketsled")
        ctx.run("make html")
        ctx.run("cp _static/* ../docs/html/_static")

    with cd("docs"):
        ctx.run("cp -r html/* .")
        ctx.run("rm -r html")
        ctx.run("rm -r doctrees")

        # Avoid the use of jekyll so that _dir works as intended.
        ctx.run("touch .nojekyll")


@task
def open_doc(ctx):
    pth = os.path.abspath("docs/index.html")
    webbrowser.open("file://" + pth)


@task
def version_check(ctx):
    with open("setup.py", "r") as f:
        setup_version = None
        for l in f.readlines():
            if "version = " in l:
                setup_version = l.split(" ")[-1]
                setup_version = setup_version.replace('"', "").replace("\n", "")

    if setup_version is None:
        raise IOError("Could not parse setup.py for version.")

    if __version__ == setup_version:
        print("Setup and init versions match eachother.")
        today = datetime.date.today().strftime("%Y.%-m.%-d")
        if today != __version__:
            raise ValueError(f"The version {__version__} does not match "
                             f"the date!")
        else:
            print("Version matches the date.")
    else:
        raise ValueError(f"There is a mismatch in the date between the "
                         f"rocketsled __init__ and the setup. Please "
                         f"make sure they are the same."
                         f"\n DIFF: {__version__}, {setup_version}")


@task
def format_project(ctx):
    ctx.run("isort --recursive rocketsled")
    ctx.run("black rocketsled")
    ctx.run("flake8 rocketsled")


@task
def update_changelog(ctx):
    version_check(ctx)
    ctx.run('github_changelog_generator hackingmaterials/rocketsled')


@task
def publish(ctx):
    version_check(ctx)
    ctx.run("rm -r dist build", warn=True)
    ctx.run("python3 setup.py sdist bdist_wheel")
    ctx.run("twine upload dist/*")


@task
def release(ctx):
    version_check(ctx)
    payload = {
        "tag_name": "v" + __version__,
        "target_commitish": "master",
        "name": "v" + __version__,
        "body": "",
        "draft": False,
        "prerelease": False
    }
    response = requests.post(
        "https://api.github.com/repos/hackingmaterials/rocketsled/releases",
        data=json.dumps(payload),
        headers={
            "Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]})
    print(response.text)
