# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import os
import json
import webbrowser
import requests
from invoke import task
from rocketsled import __version__
from monty.os import cd


"""
Deployment file to facilitate releases.
"""

__author__ = "Shyue Ping Ong, Anubhav Jain"
__email__ = "ongsp@ucsd.edu"
__date__ = "Sep 1, 2014"


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
def update_doc(ctx):
    make_doc(ctx)
    with cd("docs"):
        ctx.run("git add .")
        ctx.run("git commit -a -m \"Update to v{}\"".format(__version__))
        ctx.run("git push")


@task
def publish(ctx):
    ctx.run("rm dist/*.*", warn=True)
    ctx.run("python3 setup.py sdist bdist_wheel")
    ctx.run("twine upload dist/*")

@task
def update_changelog(ctx):
    ctx.run('github_changelog_generator hackingmaterials/rocketsled')

@task
def release(ctx):
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
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]})
    print(response.text)

@task
def open_doc(ctx):
    pth = os.path.abspath("docs/index.html")
    webbrowser.open("file://" + pth)
